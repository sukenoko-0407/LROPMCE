from dataclasses import dataclass, field
from typing import Dict, List, Any
from ..nodes import LeafNode
from .path import PathToken
from ..types import LeafKey
import time

@dataclass
class PendingEntry:
    leaf: LeafNode
    waiters: List[PathToken]
    enqueued_at: float

class PendingManager:
    def __init__(self, flush_threshold: int):
        self.pending_items: Dict[LeafKey, PendingEntry] = {}
        self.flush_threshold = flush_threshold
        
    def enqueue(self, leaf: LeafNode, token: PathToken):
        key = (leaf.leaf_smiles, leaf.depth_action)
        if key in self.pending_items:
            self.pending_items[key].waiters.append(token)
        else:
            self.pending_items[key] = PendingEntry(
                leaf=leaf,
                waiters=[token],
                enqueued_at=time.time()
            )
            
    def should_flush(self) -> bool:
        return len(self.pending_items) >= self.flush_threshold

    def pop_all(self) -> List[PendingEntry]:
        items = list(self.pending_items.values())
        self.pending_items.clear()
        return items
