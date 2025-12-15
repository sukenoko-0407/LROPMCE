from typing import List
from ..tree import MCTSTree
from ..nodes import BranchNode, LeafNode, ActionStats
import logging

logger = logging.getLogger(__name__)

def merge_trees(trees: List[MCTSTree]) -> MCTSTree:
    """
    Merges multiple MCTS trees into one.
    - N, W are summed.
    - Inflight is summed (though expected to be near 0).
    - Unions branches and leaves.
    """
    if not trees:
        raise ValueError("No trees to merge")
        
    base_tree = trees[0]
    # If only one, return copy to be safe? Or just return it.
    if len(trees) == 1:
        return base_tree
        
    # Check checkpoint ID
    ref_ckpt = base_tree.checkpoint_id
    for t in trees[1:]:
        if t.checkpoint_id != ref_ckpt:
            logger.warning(f"Merging trees with different checkpoint IDs: {ref_ckpt} vs {t.checkpoint_id}")

    # Helper to merge ActionStats
    def merge_action_stats(target: ActionStats, source: ActionStats):
        target.N += source.N
        target.W += source.W
        target.inflight += source.inflight
        if source.child_leaf and not target.child_leaf:
            target.child_leaf = source.child_leaf

    # Merge Branches
    for t in trees[1:]:
        for b_key, b_node in t.branches.items():
            if b_key not in base_tree.branches:
                # Copy node (shallow copy fine?)
                # We need deep copy of action_stats dict though
                new_node = BranchNode(
                    branch_smiles=b_node.branch_smiles,
                    depth_action=b_node.depth_action,
                    is_terminal=b_node.is_terminal,
                    mol_props_branch=b_node.mol_props_branch,
                    legal_actions=b_node.legal_actions,
                    priors_legal=b_node.priors_legal,
                    N=b_node.N,
                    W=b_node.W,
                    action_stats={k: ActionStats(v.N, v.W, v.inflight, v.child_leaf) for k,v in b_node.action_stats.items()},
                    parent_ref=b_node.parent_ref
                )
                base_tree.branches[b_key] = new_node
            else:
                # Merge stats
                target_node = base_tree.branches[b_key]
                target_node.N += b_node.N
                target_node.W += b_node.W
                for act_id, stats in b_node.action_stats.items():
                    if act_id not in target_node.action_stats:
                        target_node.action_stats[act_id] = ActionStats(stats.N, stats.W, stats.inflight, stats.child_leaf)
                    else:
                        merge_action_stats(target_node.action_stats[act_id], stats)

        # Merge Leaves
        for l_key, l_node in t.leaves.items():
             if l_key not in base_tree.leaves:
                 base_tree.leaves[l_key] = l_node # Shallow copy ok for LeafNode
             else:
                 # Update status priority: done > pending > ready > not_ready
                 target_leaf = base_tree.leaves[l_key]
                 if l_node.leaf_calc == "done":
                     target_leaf.leaf_calc = "done"
                     target_leaf.value = l_node.value
                 elif l_node.leaf_calc == "pending" and target_leaf.leaf_calc != "done":
                     target_leaf.leaf_calc = "pending"
                 # merge children_branches list (unique)
                 existing_children = set(target_leaf.children_branches)
                 for child in l_node.children_branches:
                     if child not in existing_children:
                         target_leaf.children_branches.append(child)
                         existing_children.add(child)

    return base_tree
