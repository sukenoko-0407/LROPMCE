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
