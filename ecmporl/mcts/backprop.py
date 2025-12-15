from ..tree import MCTSTree
from .path import PathToken

def backpropagate(tree: MCTSTree, path: PathToken, value: float) -> None:
    """
    Update N, W for all edges in the path.
    Also update inflight counters (decrement) - assumed handled by caller or here?
    The spec says: "inflight -= 1 (on path)" then "backprop(path, R)".
    Let's handle only N, W updates here for clarity, or both.
    Usually inflight is decremented when *leaving* the pending state.
    Let's assume the caller handles inflight decrement if they incremented it.
    Actually, to keep it safe, let's update N, W here.
    """
    for branch_key, action_id in path.edges:
        branch = tree.branches.get(branch_key)
        if branch:
            # Update Node stats
            # Spec 12.1: "BranchNode.N/W (state stats)"
            branch.N += 1
            branch.W += value
            
            # Update Action stats
            # Spec 12.1: "ActionStats.N/W (action stats)"
            if action_id in branch.action_stats:
                stats = branch.action_stats[action_id]
                stats.N += 1
                stats.W += value
