import sys
import os
import numpy as np

# Ensure local package is found
sys.path.append(os.getcwd())

from ecmporl.config import SearchConfig, ConstraintConfig, RewardConfig
from ecmporl.fragments.library import FragmentLibrary
from ecmporl.tree import MCTSTree
from ecmporl.mcts.generation import StepByStepGenerator
from ecmporl.nodes import BranchNode
from ecmporl.smiles.props import measure_mol_props
from ecmporl.fragments.legal import get_legal_actions

def test_pruning():
    print("Testing MCTSTree.prune_to_subtree...")
    initial_smiles = "*C"
    root_key = (initial_smiles, 0)
    tree = MCTSTree(checkpoint_id="test", root=root_key)
    
    # Add dummy root branch
    root_node = BranchNode(
        branch_smiles=initial_smiles,
        depth_action=0,
        is_terminal=False,
        mol_props_branch={},
        legal_actions=np.array([1, 2])
    )
    tree.add_branch(root_node)
    
    # Add child leaf and next branch
    from ecmporl.nodes import LeafNode, ActionStats
    leaf_smi = "CC*"
    leaf_node = LeafNode(leaf_smi, 1, "done", False, 1.0, {})
    tree.add_leaf(leaf_node)
    root_node.action_stats[1] = ActionStats(N=10, child_leaf=(leaf_smi, 1))
    
    next_branch_smi = "CC(C)*"
    next_branch_key = (next_branch_smi, 1)
    next_branch = BranchNode(next_branch_smi, 1, False, {}, np.array([3]))
    tree.add_branch(next_branch)
    leaf_node.children_branches = [next_branch_key]
    
    # Add an unreachable branch
    junk_key = ("Junk*", 0)
    junk_branch = BranchNode("Junk*", 0, False, {}, np.array([9]))
    tree.add_branch(junk_branch)
    
    print(f"Before prune: {len(tree.branches)} branches, {len(tree.leaves)} leaves")
    assert len(tree.branches) == 3
    
    # Prune to next_branch
    tree.prune_to_subtree(next_branch_key)
    
    print(f"After prune: {len(tree.branches)} branches, {len(tree.leaves)} leaves")
    assert len(tree.branches) == 1
    assert len(tree.leaves) == 0 # Leaf was parent, so it's gone
    assert tree.root == next_branch_key
    print("Pruning test passed!")

def test_generation():
    print("\nTesting StepByStepGenerator.generate_one...")
    
    # Configs
    s_config = SearchConfig(
        algorithm="uct",
        max_depth=3,
        max_simulations=10,
        flush_threshold=5
    )
    c_config = ConstraintConfig()
    r_config = RewardConfig()
    
    if not os.path.exists("primary_elem_lib.csv"):
        # Create a tiny dummy lib if it doesn't exist
        with open("primary_elem_lib.csv", "w") as f:
            f.write("smiles,name\n*C,methyl\n*CC,ethyl\n")
            
    frag_lib = FragmentLibrary("primary_elem_lib.csv")
    initial_smiles = "*C"
    
    root_props = measure_mol_props(initial_smiles)
    legal = get_legal_actions(root_props, frag_lib, c_config)
    
    root_node = BranchNode(
        branch_smiles=initial_smiles,
        depth_action=0,
        is_terminal=False,
        mol_props_branch=root_props,
        legal_actions=legal
    )
    
    tree = MCTSTree(checkpoint_id="gen_test", root=(initial_smiles, 0))
    tree.add_branch(root_node)
    
    rng = np.random.Generator(np.random.PCG64(42))
    generator = StepByStepGenerator(tree, s_config, c_config, r_config, frag_lib, None, rng)
    
    smiles = generator.generate_one(n_simulations_per_step=10, temperature=1.0)
    print(f"Generated SMILES: {smiles}")
    assert isinstance(smiles, str)
    assert len(smiles) > 0
    print("Generation test passed!")

if __name__ == "__main__":
    try:
        test_pruning()
        test_generation()
        print("\nAll tests passed successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
