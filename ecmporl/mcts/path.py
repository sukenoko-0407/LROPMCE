from dataclasses import dataclass
from typing import Tuple
from ..types import BranchKey, LeafKey

@dataclass(frozen=True)
class PathToken:
    # Tuple of (branch_key, action_id)
    edges: Tuple[Tuple[BranchKey, int], ...]
    leaf_key: LeafKey
