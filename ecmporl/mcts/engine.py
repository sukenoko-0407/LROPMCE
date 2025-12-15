import time
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, List
from ..tree import MCTSTree
from ..config import SearchConfig, ConstraintConfig
from ..fragments.library import FragmentLibrary
from ..model.inference import ModelInference
from ..eval.evaluator import LeafBatchEvaluator, ParallelEvaluator, SyncEvaluator
from ..eval.reward import RewardCalculator, aggregate_rewards

from .pending import PendingManager
from .simulation import Simulation
from .backprop import backpropagate

logger = logging.getLogger(__name__)

def run_search_session(
    config: SearchConfig,
    constraint_config: ConstraintConfig,
    reward_config: "RewardConfig",
    frag_lib: FragmentLibrary,
    initial_smiles: str,
    model_inference: Optional[ModelInference],
    seed: int = None
) -> MCTSTree:
    """
    Runs MCTS search session(s).
    If config.num_search_workers > 1, runs parallel independent searches and merges results.
    """
    if seed is None:
        seed = config.seed

    num_workers = config.num_search_workers
    
    if num_workers <= 1:
        return _run_single_search_session(
            config, constraint_config, reward_config, frag_lib, 
            initial_smiles, model_inference, seed
        )
    
    # Parallel Execution
    # Generate seeds
    rng = np.random.Generator(np.random.PCG64(seed))
    seeds = rng.integers(0, 2**32, size=num_workers)
    
    # We need to be careful about pickling model_inference.
    # If it's a PyTorch model on GPU, this might fail with "spawn".
    # For now, we assume CPU or safe pickling. 
    # If model_inference is None, it's safe.
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            # Submit task
            f = executor.submit(
                _run_single_search_session,
                config,
                constraint_config,
                reward_config,
                frag_lib,
                initial_smiles,
                model_inference,
                int(seeds[i])
            )
            futures.append(f)
            
        trees = [f.result() for f in futures]
        
    if not trees:
        # Should not happen
        return _run_single_search_session(
             config, constraint_config, reward_config, frag_lib, 
             initial_smiles, model_inference, seed
        )

    # Merge into the first tree
    main_tree = trees[0]
    for other_tree in trees[1:]:
        main_tree.merge(other_tree)
        
    return main_tree

def _run_single_search_session(
    config: SearchConfig,
    constraint_config: ConstraintConfig,
    reward_config: "RewardConfig",
    frag_lib: FragmentLibrary,
    initial_smiles: str,
    model_inference: Optional[ModelInference],
    seed: int
) -> MCTSTree:
    """
    Runs a single MCTS search session (worker).
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    
    # Init Tree
    # Need initial props
    # Root branch
    from ..smiles.props import measure_mol_props
    from ..fragments.legal import get_legal_actions
    from ..nodes import BranchNode
    
    # Root setup
    root_props = measure_mol_props(initial_smiles)
    legal = get_legal_actions(root_props, frag_lib, constraint_config)
    
    priors = None
    if config.algorithm == "puct" and model_inference:
        pol, _ = model_inference.predict_priors_and_value([initial_smiles])
        if pol[0] is not None:
             logits = pol[0][legal] / config.tau_policy
             exps = np.exp(logits - np.max(logits))
             priors = exps / exps.sum()

    root_node = BranchNode(
        branch_smiles=initial_smiles,
        depth_action=0,
        is_terminal=(len(legal)==0),
        mol_props_branch=root_props,
        legal_actions=legal,
        priors_legal=priors
    )
    
    tree = MCTSTree(
        checkpoint_id="init", # Placeholder
        root=(initial_smiles, 0)
    )
    tree.add_branch(root_node)
    
    # Components
    pending_manager = PendingManager(config.flush_threshold)
    sim = Simulation(tree, config, constraint_config, frag_lib, pending_manager, model_inference, rng)
    
    # Evaluator
    rew_calc = RewardCalculator(reward_config)
    # Use SyncEvaluator for process-local evaluation (batched if possible)
    evaluator = SyncEvaluator(rew_calc)
    
    # Loop
    sim_count = 0
    start_time = time.time()
    
    while sim_count < config.max_simulations:
        # Run one
        prev_pending = len(pending_manager.pending_items)
        sim.run_one()
        sim_count += 1
        
        # Flush check
        if pending_manager.should_flush():
             _flush_pending(pending_manager, evaluator, tree)
        elif len(pending_manager.pending_items) > 0 and len(pending_manager.pending_items) == prev_pending:
             # Deadlock detection: We have pending items but run_one() added nothing.
             # This happens if all legal actions are inflight (and excluded).
             # We must flush to proceed.
             _flush_pending(pending_manager, evaluator, tree)

    # Final flush
    _flush_pending(pending_manager, evaluator, tree)
    
    return tree

def _flush_pending(pm: PendingManager, evaluator, tree: MCTSTree):
    entries = pm.pop_all()
    if not entries:
        return
        
    leaf_smiles = [e.leaf.leaf_smiles for e in entries]
    rewards = evaluator.evaluate(leaf_smiles)
    
    for entry, reward in zip(entries, rewards):
        leaf = entry.leaf
        leaf.value = reward
        leaf.leaf_calc = "done"
        
        # Backprop for all waiters
        for token in entry.waiters:
            # Decrement inflight? (Handled implicitly or explicit update?)
            # Logic: Inflight was part of Search decision (Q_eff).
            # Once value confirmed, we update N/W and inflight effect is gone (if we decr).
            # Spec 11: "評価確定時に対応する inflight -= 1 を行い、その後に Backpropagate"
            # We must decrement.
            _decrement_inflight(tree, token.edges)
            backpropagate(tree, token, reward)

def _decrement_inflight(tree, edges):
    for b_key, a_id in edges:
        b = tree.get_branch(b_key)
        if b and a_id in b.action_stats:
            b.action_stats[a_id].inflight = max(0, b.action_stats[a_id].inflight - 1)
