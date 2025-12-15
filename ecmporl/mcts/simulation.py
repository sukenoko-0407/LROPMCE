import numpy as np
from typing import Optional, List, Any
import logging

from ..nodes import BranchNode, LeafNode, ActionStats
from ..tree import MCTSTree
from ..config import SearchConfig, ConstraintConfig
from ..types import BranchKey, LeafKey
from .path import PathToken
from .pending import PendingManager
from .selection import select_action
from .leaf_to_branch import select_next_branch
from .backprop import backpropagate

from ..smiles.ops import combine_smiles, hydrogen_replace, canonicalize
from ..smiles.props import measure_mol_props
from ..smiles.alerts import alert_ok_mol, alert_ok_elem
from ..fragments.library import FragmentLibrary
from ..fragments.legal import get_legal_actions
from ..model.inference import ModelInference

logger = logging.getLogger(__name__)

class Simulation:
    def __init__(
        self,
        tree: MCTSTree,
        config: SearchConfig,
        constraint_config: ConstraintConfig,
        frag_lib: FragmentLibrary,
        pending_manager: PendingManager,
        model_inference: Optional[ModelInference], # For PUCT priors
        rng: np.random.Generator
    ):
        self.tree = tree
        self.config = config
        self.constraint_config = constraint_config
        self.frag_lib = frag_lib
        self.pending_manager = pending_manager
        self.model = model_inference
        self.rng = rng

    def run_one(self) -> None:
        """
        Executes one simulation step.
        1. Selection (Leaf-ward)
        2. Expansion (Action application)
        3. Leaf Handling (Evaluated / Pending / Not Ready -> Next Branch)
        """
        # Start from root
        current_branch_key = self.tree.root
        current_branch = self.tree.get_branch(current_branch_key)
        
        path_edges = []
        
        # 1. Selection
        while True:
            # Check if terminal
            if current_branch.is_terminal:
                # Reached terminal branch. Backprop 0? Or what value?
                # Spec doesn't explicitly say. Usually implies dead end or done.
                # If terminal, it can't expand.
                # Backprop current stored value? or 0.
                backpropagate(self.tree, PathToken(tuple(path_edges), ("", 0)), 0.0)
                return

            action_id, is_new = select_action(current_branch, self.config, self.rng)
            
            if action_id == -1:
                # No legal actions? Dead end.
                current_branch.is_terminal = True
                backpropagate(self.tree, PathToken(tuple(path_edges), ("", 0)), 0.0)
                return

            # Add to path
            path_edges.append((current_branch_key, action_id))

            # Get Stats
            stats = current_branch.action_stats.get(action_id)
            if not stats:
                # Initialize stats if new
                stats = ActionStats()
                current_branch.action_stats[action_id] = stats
            
            # If child already exists, traverse.
            if stats.child_leaf:
                child_leaf = self.tree.get_leaf(stats.child_leaf)
                # Check outcome
                if child_leaf.leaf_calc == "done":
                    # FIX: If we came via Pending/Flush, we might not have expanded branches yet.
                    if not child_leaf.children_branches and not child_leaf.is_terminal:
                         self._expand_branches_from_leaf(child_leaf, path_edges)
                         # If still no branches (and marked terminal), backprop happens in expand.
                         # But expand returns None, so we need to check terminal again.
                         if child_leaf.is_terminal:
                             return

                    # Find next branch
                    next_branch = self._select_branch_from_leaf(child_leaf)
                    if next_branch:
                        current_branch = next_branch
                        current_branch_key = (next_branch.branch_smiles, next_branch.depth_action)
                        continue
                    else:
                        # Dead end or leaf-only terminal
                        backpropagate(self.tree, PathToken(tuple(path_edges), child_leaf), child_leaf.value)
                        return

                elif child_leaf.leaf_calc == "pending":
                    # Pending. Add to waiters.
                    # NOTE: Spec 10.3 says exclude pending from selection.
                    # If we are here, select_action logic FAILED to exclude it, 
                    # OR we are allowed to wait.
                    # My selection.py excludes if inflight > 0.
                    # If inflight=0 but pending? (Should not happen if flush logic creates inflight).
                    # Let's handle it as "Waiting".
                    token = PathToken(tuple(path_edges), (child_leaf.leaf_smiles, child_leaf.depth_action))
                    self.pending_manager.enqueue(child_leaf, token)
                    # Add inflight
                    self._add_inflight(path_edges)
                    return
                
                elif child_leaf.leaf_calc == "not_ready":
                     # Should have expanded to Branch already.
                     # Treated same as "done" but value is None implies intermediate.
                     # Proceed to branch selection.
                    next_branch = self._select_branch_from_leaf(child_leaf)
                    if next_branch:
                        current_branch = next_branch
                        current_branch_key = (next_branch.branch_smiles, next_branch.depth_action)
                        continue
                    else:
                         # Dead end
                         backpropagate(self.tree, PathToken(tuple(path_edges), ("",0)), 0.0)
                         return

                elif child_leaf.leaf_calc == "ready":
                     # Ready but not pending? -> New ready node.
                     # Enqueue.
                    token = PathToken(tuple(path_edges), (child_leaf.leaf_smiles, child_leaf.depth_action))
                    child_leaf.leaf_calc = "pending"
                    self.pending_manager.enqueue(child_leaf, token)
                    self._add_inflight(path_edges)
                    return
            
            else:
                # New expansion
                self._expand_action(current_branch, action_id, path_edges)
                return

    def _expand_action(self, branch: BranchNode, action_id: int, path_edges: List):
        # 1. combine_smiles
        frag_smiles = self.frag_lib.get_fragment_smiles(action_id)
        try:
            leaf_smiles = combine_smiles(branch.branch_smiles, frag_smiles)
        except ValueError:
             # Combination failed. Terminal / Value 0.
             self._mark_action_terminal(branch, action_id, path_edges)
             return

        # Create LeafNode
        # Check depth
        new_depth = branch.depth_action + 1
        
        # Check Alert Mol
        if  alert_ok_mol(leaf_smiles) == 0:
            # NG -> Value 0
             # print(f"DEBUG: Rejected by alert_ok_mol: {leaf_smiles}")
             self._create_terminal_leaf(branch, action_id, leaf_smiles, new_depth, path_edges, value=0.0)
             return
             
        # Create Node
        props = measure_mol_props(leaf_smiles)
        
        leaf_node = LeafNode(
            leaf_smiles=leaf_smiles,
            depth_action=new_depth,
            leaf_calc="not_ready", # Tentative
            is_terminal=False,
            value=None,
            mol_props=props,
            children_branches=[],
            parent_ref=branch
        )
        
        # Add to Tree
        leaf_key = self.tree.add_leaf(leaf_node)
        
        # Link from parent
        branch.action_stats[action_id].child_leaf = leaf_key
        
        # Determine Status
        # 1. Max Depth
        if new_depth >= self.config.max_depth:
            leaf_node.is_terminal = True
            leaf_node.leaf_calc = "ready" # Evaluate even if terminal
        else:
            # 2. Ready Check (Min constraints)
            is_ready = self.constraint_config.validate_leaf(props)
            is_depth_ok = new_depth >= self.config.min_depth
            
            if is_ready and is_depth_ok:
                leaf_node.leaf_calc = "ready"
            else:
                leaf_node.leaf_calc = "not_ready"
                # if not is_ready: print(f"DEBUG: Leaf not ready (constraints?): Props={props}")
                # if not is_depth_ok: print(f"DEBUG: Leaf not depth ok: {new_depth} < {self.config.min_depth}")

        # Action:
        if leaf_node.leaf_calc == "ready":
            # Enqueue
            token = PathToken(tuple(path_edges), leaf_key)
            leaf_node.leaf_calc = "pending"
            self.pending_manager.enqueue(leaf_node, token)
            self._add_inflight(path_edges)
            return
            
        elif leaf_node.leaf_calc == "not_ready":
             # Immediate expansion to Branch
             # "not_ready の場合: hydrogen_replace で Branch 候補を生成し... 次へ"
             # Since this is "Expansion" step, we continue locally (recursive call or loop).
             # But run_one() loop expects to start from root? No, we are deep.
             # We should continue the expansion here.
             
             self._expand_branches_from_leaf(leaf_node, path_edges)
             return

    def _expand_branches_from_leaf(self, leaf: LeafNode, path_edges: List):
        # 1. hydrogen_replace
        try:
             candidates = hydrogen_replace(leaf.leaf_smiles)
        except ValueError:
             candidates = []
             
        # Filter alert_ok_elem
        valid_candidates = []
        for cand_smi in candidates:
             if alert_ok_elem(cand_smi):
                 valid_candidates.append(cand_smi)
        
        if not valid_candidates:
             # Leaf-only terminal.
             # print(f"DEBUG: No valid branch candidates for {leaf.leaf_smiles}. H-replace: {len(candidates)}, Alert-OK: {len(valid_candidates)}")
             leaf.is_terminal = True
             leaf.value = 0.0
             leaf.leaf_calc = "done"
             backpropagate(self.tree, PathToken(tuple(path_edges), (leaf.leaf_smiles, leaf.depth_action)), 0.0)
             return

        # Create BranchNodes
        branches = []
        for branch_smi in valid_candidates:
             # Check if exists in tree? (Transposition)
             branch_key = (branch_smi, leaf.depth_action)
             existing = self.tree.get_branch(branch_key)
             if existing:
                 branches.append(existing)
             else:
                 # Create new
                 # Calc props for filters
                 # For BranchNode props, measure_mol_props expects real molecule.
                 # Branch has dummy. Spec: "branch_equivalent_leaf_smiles" -> usually remove dummy?
                 # Or treat as is? measure_mol_props handles dummy.
                 b_props = measure_mol_props(branch_smi) 
                 # Get legal actions
                 legal = get_legal_actions(b_props, self.frag_lib, self.constraint_config)
                 
                 # Inference priors if PUCT
                 # We can batch this? But here we are creating nodes.
                 # "Branch生成時（または到達時）... legal action"
                 # Priors filling:
                 # Ideally we query model here.
                 priors_legal = None
                 if self.config.algorithm == "puct" and self.model:
                     # Get single inference (inefficient? But needed for new branch)
                     # Or defer? But selection needs it.
                     # Let's allow lazy init or do it now.
                     pol, _ = self.model.predict_priors_and_value([branch_smi])
                     if pol[0] is not None:
                         # Extract legal
                         logits = pol[0][legal]
                         # Softmax with temp
                         logits = logits / self.config.tau_policy
                         exp_l = np.exp(logits - np.max(logits))
                         priors_legal = exp_l / exp_l.sum()
                 
                 new_branch = BranchNode(
                     branch_smiles=branch_smi,
                     depth_action=leaf.depth_action,
                     is_terminal=(len(legal)==0),
                     mol_props_branch=b_props,
                     legal_actions=legal,
                     priors_legal=priors_legal,
                     parent_ref=leaf
                 )
                 self.tree.add_branch(new_branch)
                 branches.append(new_branch)
        
        leaf.children_branches = [(b.branch_smiles, b.depth_action) for b in branches]
        
        # Select next branch
        # Need priors for branches if PUCT
        branch_priors_map = None
        if self.config.algorithm == "puct" and self.model:
             # Batch value inference for branches
             smiles_list = [b.branch_smiles for b in branches]
             values = self.model.predict_values(smiles_list)
             # Softmax
             # P = softmax(V / tau)
             # values are [0,1] or similar.
             vs = np.array(values)
             vs = vs / self.config.tau_branch
             exps = np.exp(vs - np.max(vs))
             probs = exps / exps.sum()
             branch_priors_map = {b.branch_smiles: p for b, p in zip(branches, probs)}

        next_branch = select_next_branch(leaf, branches, self.config, self.rng, branch_priors_map)
        
        if next_branch:
             # Continue simulation from here?
             # Since run_one() loop is simpler if we just continue.
             # But run_one() is iteratively going down.
             # We can recursively call run_one-like logic or just return to main loop?
             # Main loop is "Selection". 
             # We are in "Expansion".
             # If we expanded to a Branch, we are now at a Branch.
             # We can effectively "jump" to that Branch and continue Selection loop.
             # BUT we need to handle the Recursion structure.
             # Let's refactor run_one() to be a single loop that handles "Current Node = Branch".
             # Expansion called from loop.
             # If expansion results in a new Branch, we want to continue the loop with that Branch.
             # Implementation:
             # _expand_action -> calls _expand_branches -> returns Next Branch.
             pass
             # Refactoring for iterative approach.
             # See below.

    def _create_terminal_leaf(self, branch, action_id, smiles, depth, path_edges, value):
        leaf = LeafNode(
            leaf_smiles=smiles,
            depth_action=depth,
            leaf_calc="done",
            is_terminal=True,
            value=value,
            mol_props={}, # Skip calc
            children_branches=[]
        )
        key = self.tree.add_leaf(leaf)
        branch.action_stats[action_id].child_leaf = key
        backpropagate(self.tree, PathToken(tuple(path_edges), key), value)

    def _add_inflight(self, path_edges):
        for b_key, a_id in path_edges:
            b = self.tree.get_branch(b_key)
            if b and a_id in b.action_stats:
                b.action_stats[a_id].inflight += 1

    def _select_branch_from_leaf(self, leaf: LeafNode) -> Optional[BranchNode]:
        """Simple wrapper to re-select or get existing branches"""
        if not leaf.children_branches:
            return None
        branches = [self.tree.get_branch(k) for k in leaf.children_branches]
        branches = [b for b in branches if b] # filter None
        if not branches:
            return None
            
        # If we need to re-calc priors?
        # Assuming cached or deterministic is fine.
        # But for PUCT we need values.
        # Assuming select_next_branch handles it or we precalc.
        # For simplicity, if branches exist, we just start UCT/PUCT selection.
        # Ideally we cache the Branch Level priors?
        # But leaf_to_branch logic is lightweight.
        
        # Need model for PUCT
        branch_priors_map = None
        if self.config.algorithm == "puct" and self.model:
             # Re-predict? Expensive.
             # Should cache on Leaf? LeafNode doesn't hold priors.
             # Let's re-predict for now or ignore. 
             # Spec says "Leaf→Branch 選択時の Value 推論は...バッチ化...必須".
             # This implies we do it whenever we select.
             smiles_list = [b.branch_smiles for b in branches]
             values = self.model.predict_values(smiles_list)
             vs = np.array(values) / self.config.tau_branch
             exps = np.exp(vs - np.max(vs))
             probs = exps / exps.sum()
             branch_priors_map = {b.branch_smiles: p for b, p in zip(branches, probs)}

        return select_next_branch(leaf, branches, self.config, self.rng, branch_priors_map)

