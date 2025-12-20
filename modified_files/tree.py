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

    def prune_to_subtree(self, new_root_key: BranchKey):
        """
        指定された new_root_key から到達不能なすべてのノードを削除する。
        BFS (幅優先探索) を用いて到達可能なノードをマークします。
        """
        reachable_branches = set()
        reachable_leaves = set()
        
        queue = [new_root_key]
        while queue:
            b_key = queue.pop(0)
            if b_key in reachable_branches:
                continue
            
            branch = self.branches.get(b_key)
            if not branch:
                continue
            
            reachable_branches.add(b_key)
            
            # ActionStats を通じて子 Leaf を辿る
            for stats in branch.action_stats.values():
                if stats.child_leaf:
                    l_key = stats.child_leaf
                    if l_key not in reachable_leaves:
                        reachable_leaves.add(l_key)
                        leaf = self.leaves.get(l_key)
                        if leaf:
                            # Leaf から子 Branch を辿る
                            for child_b_key in leaf.children_branches:
                                if child_b_key not in reachable_branches:
                                    queue.append(child_b_key)

        # 到達不能なノードを削除
        self.branches = {k: v for k, v in self.branches.items() if k in reachable_branches}
        self.leaves = {k: v for k, v in self.leaves.items() if k in reachable_leaves}
        self.root = new_root_key
