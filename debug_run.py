import sys
import os
import logging
sys.path.append(os.getcwd())
from ecmporl.config import SearchConfig, ConstraintConfig, RewardConfig
from ecmporl.fragments.library import FragmentLibrary
from ecmporl.parallel.worker import run_worker
from ecmporl.model.inference import ModelInference
from ecmporl.model.pvnet import PolicyValueNet

# Monkey patch print debugging into simulation if needed, 
# or just run and rely on existing logs if we enable DEBUG level?
# ecmporl.mcts.simulation uses logger.
# Let's configure logging to see details.

logging.basicConfig(level=logging.DEBUG)

def main():
    print("Running Debug Simulation...")
    # Configs (same as run_example)
    s_config = SearchConfig(
        algorithm="uct",
        min_depth=2,
        max_depth=5,
        max_simulations=500, # Try 500 first to see trend
        flush_threshold=16
    )
    c_config = ConstraintConfig(
        HAC_max=50,
        MW_max=600.0,
        hetero_max=10
    )
    r_config = RewardConfig()
    
    if not os.path.exists("primary_elem_lib.csv"):
        print("Error: primary_elem_lib.csv missing")
        return
        
    frag_lib = FragmentLibrary("primary_elem_lib.csv")
    model = PolicyValueNet(action_dim=frag_lib.K)
    inference = ModelInference(model)
    
    initial_smiles = "*C1CC1"
    
    tree = run_worker(
        config=s_config,
        constraint_config=c_config,
        reward_config=r_config,
        frag_lib=frag_lib,
        initial_smiles=initial_smiles,
        model_inference=inference, 
        seed=42
    )
    
    print(f"Complete. Branches: {len(tree.branches)}, Leaves: {len(tree.leaves)}")
    
    # Analyze tree
    from ecmporl.eval.analysis import tree_to_dataframes
    df_branch, df_leaf = tree_to_dataframes(tree)
    
    print("\n--- Branch DataFrame Head ---")
    print(df_branch.head())
    print("\nBranch Info:")
    pass # df_branch.info() prints to stderr/stdout depending on implementation, keep it simple
    print(df_branch.describe())

    print("\n--- Leaf DataFrame Head ---")
    print(df_leaf.head())
    print("\nLeaf Value Counts (Status):")
    print(df_leaf['status'].value_counts())

if __name__ == "__main__":
    main()
