from typing import Tuple, Dict, Any, List
import pandas as pd
from ..tree import MCTSTree

def tree_to_dataframes(tree: MCTSTree) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts the MCTS tree into two pandas DataFrames:
    1. Branch DataFrame: Contains information about all BranchNodes.
    2. Leaf DataFrame: Contains information about all LeafNodes.

    Returns:
        (df_branches, df_leaves)
    """
    
    # --- Branch Nodes ---
    branch_data = []
    for key, branch in tree.branches.items():
        # Flatten properties
        props = _flatten_props(branch.mol_props_branch, prefix="prop_")
        
        # Calculate stats
        mean_val = 0.0
        if branch.N > 0:
            mean_val = branch.W / branch.N
            
        record = {
            "smiles": branch.branch_smiles,
            "depth": branch.depth_action,
            "is_terminal": branch.is_terminal,
            "visit_count": branch.N,
            "mean_value": mean_val,
            "num_actions": len(branch.action_stats),
            **props
        }
        branch_data.append(record)
        
    df_branches = pd.DataFrame(branch_data)
    
    # --- Leaf Nodes ---
    leaf_data = []
    for key, leaf in tree.leaves.items():
        # Flatten properties
        props = _flatten_props(leaf.mol_props, prefix="prop_")
        
        record = {
            "smiles": leaf.leaf_smiles,
            "depth": leaf.depth_action,
            "status": leaf.leaf_calc,
            "is_terminal": leaf.is_terminal,
            "value": leaf.value,
            **props
        }
        leaf_data.append(record)
        
    df_leaves = pd.DataFrame(leaf_data)
    
    return df_branches, df_leaves

def _flatten_props(props: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Helper to flatten property dictionary with a prefix."""
    if not props:
        return {}
    return {f"{prefix}{k}": v for k, v in props.items()}
