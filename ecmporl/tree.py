from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .types import LeafKey, BranchKey

# Avoid circular import for type hinting if possible, or use string forward refs
# But here we import classes for usage.
from .nodes import BranchNode, LeafNode

@dataclass
class MCTSTree:
    checkpoint_id: str
    root: BranchKey
    branches: Dict[BranchKey, BranchNode] = field(default_factory=dict)
    leaves: Dict[LeafKey, LeafNode] = field(default_factory=dict)
    
    # Optional simulation records for training data extraction
    # sim_records: list = field(default_factory=list)

    def get_branch(self, key: BranchKey) -> Optional[BranchNode]:
        return self.branches.get(key)

    def get_leaf(self, key: LeafKey) -> Optional[LeafNode]:
        return self.leaves.get(key)
    
    def add_branch(self, node: BranchNode) -> BranchKey:
        key = (node.branch_smiles, node.depth_action)
        self.branches[key] = node
        return key

    def add_leaf(self, node: LeafNode) -> LeafKey:
        key = (node.leaf_smiles, node.depth_action)
        self.leaves[key] = node
        return key

    def merge(self, other: "MCTSTree"):
        """
        Merge another tree into this one.
        """
        # Merge branches
        for key, other_branch in other.branches.items():
            if key in self.branches:
                self.branches[key].merge(other_branch)
            else:
                # Naive copy. ideally deepcopy if needed, but safe if other is discarded.
                self.branches[key] = other_branch

        # Merge leaves (just union, assuming deterministic properties)
        for key, other_leaf in other.leaves.items():
            if key not in self.leaves:
                self.leaves[key] = other_leaf
