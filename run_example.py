import sys
import os
import logging

# Ensure local package is found
sys.path.append(os.getcwd())

from ecmporl.config import SearchConfig, ConstraintConfig, RewardConfig, FragmentLibraryConfig
from ecmporl.fragments.library import FragmentLibrary
from ecmporl.parallel.worker import run_worker
from ecmporl.infer.extract import extract_top_leaves
# from ecmporl.model.pvnet import PolicyValueNet # Optional, can run without model for UCT
from ecmporl.model.inference import ModelInference
from ecmporl.model.pvnet import PolicyValueNet

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing ECMPORL Verification...")
    
    # 1. Configs
    s_config = SearchConfig(
        algorithm="uct", # Test UCT first
        min_depth=2,
        max_depth=5,
        max_simulations=50,
        flush_threshold=16
    )
    
    c_config = ConstraintConfig(
        HAC_max=50,
        MW_max=600.0,
        hetero_max=10
    )
    
    r_config = RewardConfig()
    
    # 2. Fragment Library
    # Ensure csv exists
    if not os.path.exists("primary_elem_lib.csv"):
        print("Error: primary_elem_lib.csv not found.")
        return
        
    frag_lib = FragmentLibrary("primary_elem_lib.csv")
    print(f"Fragment Library loaded: {frag_lib.K} fragments")
    
    # 3. Model (Optional for UCT, but let's test instantiation)
    # Using small dummy model for speed
    print("Creating dummy PolicyValueNet...")
    model = PolicyValueNet(action_dim=frag_lib.K)
    inference = ModelInference(model)
    
    # 4. Run Worker
    initial_smiles = "*C1CC1" # Cyclobutane radical as root? Or just *C?
    # Library has *C, *N etc.
    # Root must have * for BranchNode.
    print(f"Starting search from: {initial_smiles}")
    
    tree = run_worker(
        config=s_config,
        constraint_config=c_config,
        reward_config=r_config,
        frag_lib=frag_lib,
        initial_smiles=initial_smiles,
        model_inference=inference, 
        seed=42
    )
    
    print(f"Search complete. Tree branches: {len(tree.branches)}, Leaves: {len(tree.leaves)}")
    
    # 5. Extract
    top_leaves = extract_top_leaves(tree, frag_lib, top_k=5)
    
    print("\nTop 5 Leaves:")
    for idx, res in enumerate(top_leaves):
        print(f"{idx+1}. Value: {res.value:.4f}, Visits: {res.visit_count}, Depth: {res.depth}")
        print(f"   SMILES: {res.leaf_smiles}")
        print(f"   Props: {res.mol_props}")

if __name__ == "__main__":
    main()
