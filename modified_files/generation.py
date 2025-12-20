import numpy as np
import logging
from typing import List, Optional, Tuple

from ..tree import MCTSTree
from ..config import SearchConfig, ConstraintConfig, RewardConfig
from ..fragments.library import FragmentLibrary
from ..model.inference import ModelInference
from .simulation import Simulation
from .engine import _flush_pending
from .pending import PendingManager
from ..eval.evaluator import SyncEvaluator
from ..eval.reward import RewardCalculator

logger = logging.getLogger(__name__)

class StepByStepGenerator:
    def __init__(
        self,
        tree: MCTSTree,
        config: SearchConfig,
        constraint_config: ConstraintConfig,
        reward_config: RewardConfig,
        frag_lib: FragmentLibrary,
        model_inference: Optional[ModelInference],
        rng: np.random.Generator
    ):
        self.tree = tree
        self.config = config
        self.constraint_config = constraint_config
        self.reward_config = reward_config
        self.frag_lib = frag_lib
        self.model_inference = model_inference
        self.rng = rng

    def generate_one(self, n_simulations_per_step: int, temperature: float = 1.0) -> str:
        """
        Executes step-by-step generation until a terminal node is reached or max depth.
        Returns the finalized SMILES.
        """
        rew_calc = RewardCalculator(self.reward_config)
        evaluator = SyncEvaluator(rew_calc)

        step = 0
        while True:
            current_branch = self.tree.get_branch(self.tree.root)
            if not current_branch:
                logger.error(f"Root branch {self.tree.root} not found in tree.")
                break
                
            if current_branch.is_terminal:
                logger.info(f"Reached terminal branch at step {step}: {current_branch.branch_smiles}")
                break

            # 1. Run simulations to build statistics for the current root
            pending_manager = PendingManager(self.config.flush_threshold)
            sim = Simulation(
                self.tree, self.config, self.constraint_config, 
                self.frag_lib, pending_manager, self.model_inference, self.rng
            )
            
            # Progress simulations
            for _ in range(n_simulations_per_step):
                sim.run_one()
                if pending_manager.should_flush():
                    _flush_pending(pending_manager, evaluator, self.tree)
            
            # Final flush for this step
            _flush_pending(pending_manager, evaluator, self.tree)

            # 2. Stochastic action selection
            action_id = self._select_next_action(current_branch, temperature)
            if action_id == -1:
                logger.warning(f"No valid action found for branch {current_branch.branch_smiles} at step {step}")
                break

            # 3. Decision & Move
            stats = current_branch.action_stats.get(action_id)
            if not (stats and stats.child_leaf):
                logger.error(f"Selected action {action_id} has no child leaf.")
                break
            
            child_leaf = self.tree.get_leaf(stats.child_leaf)
            logger.info(f"Step {step}: Selected action {action_id} -> {child_leaf.leaf_smiles}")

            if child_leaf.is_terminal:
                # Finished or dead end
                self.tree.root = (child_leaf.leaf_smiles, child_leaf.depth_action)
                break
            
            # Find next branch candidate from the leaf (expansion or re-selection)
            # Re-use simulation's leaf_to_branch logic
            next_branch = sim._select_branch_from_leaf(child_leaf)
            if not next_branch:
                logger.warning(f"No branch found after leaf {child_leaf.leaf_smiles}")
                break
            
            # 4. Root update and Pruning
            new_root_key = (next_branch.branch_smiles, next_branch.depth_action)
            logger.info(f"Step {step}: Moving to Branch {new_root_key[0]}")
            
            # Prune tree to keep only the subtree starting from the new root
            self.tree.prune_to_subtree(new_root_key)
            
            step += 1
            if step > self.config.max_depth * 2: # Safety break
                break
                
        return self.tree.root[0]

    def _select_next_action(self, branch, tau: float) -> int:
        """
        Select an action based on visit counts N with temperature tau.
        """
        legal_actions = branch.legal_actions
        # Filter actions that actually have stats
        available_actions = [a for a in legal_actions if a in branch.action_stats]
        
        if not available_actions:
            return -1
            
        counts = np.array([branch.action_stats[a].N for a in available_actions], dtype=float)
        
        if tau <= 0.05:
            # Greedy: select max N
            idx = np.argmax(counts)
            return int(available_actions[idx])
        
        # P ~ N^(1/tau)
        # Handle cases with 0 counts safely or log domain
        # Using a small epsilon to avoid divide by zero if N is somehow very low (should be >=1 though)
        weights = np.power(counts, 1.0 / tau)
        sum_weights = weights.sum()
        
        if sum_weights == 0:
            # Fallback to random among visited actions
            return int(self.rng.choice(available_actions))
            
        probs = weights / sum_weights
        return int(self.rng.choice(available_actions, p=probs))
