from dataclasses import dataclass
from typing import List, Optional
from ..tree import MCTSTree
from ..fragments.library import FragmentLibrary

@dataclass
class LeafResult:
    leaf_smiles: str
    value: float
    visit_count: int
    depth: int
    mol_props: dict

def extract_top_leaves(
    tree: MCTSTree,
    frag_lib: FragmentLibrary,
    top_k: int = 10,
    value_threshold: Optional[float] = None,
    visit_threshold: Optional[int] = None
) -> List[LeafResult]:
    """
    Extract promising molecules (leaves) from the tree.
    Iterates over all branches and their action stats to find 'done' leaves.
    Returns LeafResult sorted by value.
    """
    results = []
    
    # Iterate all BranchNodes
    for b_node in tree.branches.values():
        for act_id, stats in b_node.action_stats.items():
            if not stats.child_leaf:
                continue
                
            leaf_key = stats.child_leaf
            leaf_node = tree.get_leaf(leaf_key)
            
            if not leaf_node or leaf_node.leaf_calc != "done" or leaf_node.value is None:
                continue
                
            # Filters
            if value_threshold is not None and leaf_node.value < value_threshold:
                continue
            if visit_threshold is not None and stats.N < visit_threshold:
                continue
                
            results.append(LeafResult(
                leaf_smiles=leaf_node.leaf_smiles,
                value=leaf_node.value,
                visit_count=stats.N,
                depth=leaf_node.depth_action,
                mol_props=leaf_node.mol_props
            ))
            
    # Sort by value desc
    results.sort(key=lambda x: x.value, reverse=True)
    
    if top_k:
        results = results[:top_k]
        
    return results
