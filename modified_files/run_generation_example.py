import sys
import os
import logging
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
from ecmporl.model.pvnet import PolicyValueNet
from ecmporl.model.inference import ModelInference

# Configure logging to see the steps
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_diversity_demo():
    print("=== ECMPORL Step-by-Step Diversified Generation Demo ===")
    
    # 1. Configs
    s_config = SearchConfig(
        algorithm="uct", 
        max_depth=5,
        max_simulations=100, # total sims if not step-by-step
        flush_threshold=20
    )
    c_config = ConstraintConfig(HAC_max=30)
    r_config = RewardConfig()
    
    if not os.path.exists("primary_elem_lib.csv"):
        print("Error: primary_elem_lib.csv not found.")
        return
        
    frag_lib = FragmentLibrary("primary_elem_lib.csv")
    
    # Optional Model
    model = PolicyValueNet(action_dim=frag_lib.K)
    inference = ModelInference(model)
    
    initial_smiles = "*C1CC1" # Root
    
    # We want to generate 3 different molecules from the same root
    results = []
    
    for i in range(3):
        print(f"\n--- Generating Molecule #{i+1} ---")
        
        # Initialize fresh tree for each run
        root_props = measure_mol_props(initial_smiles)
        legal = get_legal_actions(root_props, frag_lib, c_config)
        root_node = BranchNode(
            branch_smiles=initial_smiles,
            depth_action=0,
            is_terminal=False,
            mol_props_branch=root_props,
            legal_actions=legal
        )
        tree = MCTSTree(checkpoint_id=f"demo_{i}", root=(initial_smiles, 0))
        tree.add_branch(root_node)
        
        rng = np.random.Generator(np.random.PCG64(42 + i)) # Different seed for each run
        generator = StepByStepGenerator(tree, s_config, c_config, r_config, frag_lib, inference, rng)
        
        # temperature=1.2 to encourage diversity
        final_smiles = generator.generate_one(n_simulations_per_step=50, temperature=1.2)
        results.append(final_smiles)
        
        print(f"Final Tree Size: {len(tree.branches)} branches, {len(tree.leaves)} leaves")
        print(f"Resulting SMILES: {final_smiles}")

    print("\n=== Summary of Generated Molecules ===")
    for i, smi in enumerate(results):
        print(f"{i+1}: {smi}")

if __name__ == "__main__":
    run_diversity_demo()
