from dataclasses import dataclass, field
from typing import Literal, Any, Dict, List, Optional
import numpy as np
from .types import LeafKey, BranchKey

@dataclass
class LeafNode:
    leaf_smiles: str
    depth_action: int
    leaf_calc: Literal["not_ready", "ready", "pending", "done"]
    is_terminal: bool
    value: Optional[float]
    mol_props: Dict[str, Any]  # HAC, MW, etc.
    children_branches: List[BranchKey] = field(default_factory=list)
    parent_ref: Any = None # Optional reference to parent BranchNode (or None if root)

@dataclass
class ActionStats:
    N: int = 0
    W: float = 0.0
    inflight: int = 0
    child_leaf: Optional[LeafKey] = None

@dataclass
class BranchNode:
    branch_smiles: str
    depth_action: int
    is_terminal: bool
    mol_props_branch: Dict[str, Any]
    legal_actions: np.ndarray  # int array of legal action_ids
    priors_legal: Optional[np.ndarray] = None # float array, sum to 1.0 (PUCT)
    
    # Node-level stats
    N: int = 0
    W: float = 0.0
    
    # Action-level stats: action_id -> ActionStats
    action_stats: Dict[int, ActionStats] = field(default_factory=dict)
    
    parent_ref: Any = None # Optional reference to parent LeafNode (or None if root)

    def get_q_eff(self, action_id: int, vloss: float = 1.0) -> float:
        stats = self.action_stats.get(action_id)
        if not stats:
            return 0.0 # Should ideally handle unvisited separately, but Q=0 is per spec
        n_eff = stats.N + stats.inflight
        if n_eff < 1:
            return 0.0 # Or appropriate init value
        return (stats.W - vloss * stats.inflight) / n_eff

    def merge(self, other: "BranchNode"):
        """
        Merge stats from another BranchNode (same branch_smiles).
        Sum N and W. Recursively merge action_stats.
        """
        if self.branch_smiles != other.branch_smiles:
            raise ValueError("Cannot merge BranchNodes with different smiles")
        
        self.N += other.N
        self.W += other.W
        
        for aid, other_stats in other.action_stats.items():
            if aid not in self.action_stats:
                # Copy if new
                self.action_stats[aid] = ActionStats(
                    N=other_stats.N,
                    W=other_stats.W,
                    inflight=0, # Inflight is transient, reset to 0 for merged result
                    child_leaf=other_stats.child_leaf
                )
            else:
                # Merge existing
                self_stats = self.action_stats[aid]
                self_stats.N += other_stats.N
                self_stats.W += other_stats.W
                # Ignore inflight/terminals merge logic conflicts for now; assume consistent
                if self_stats.child_leaf is None and other_stats.child_leaf is not None:
                     self_stats.child_leaf = other_stats.child_leaf
