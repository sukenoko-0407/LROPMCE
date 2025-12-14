import numpy as np
from typing import List, Optional
from ..nodes import LeafNode, BranchNode
from ..config import SearchConfig
# We need inference capability here for PUCT mode ("Value推論のバッチ化")
# But passing the 'model' to this logic might be cleaner if done via an interface or callback.

def select_next_branch(
    leaf: LeafNode,
    branches: List[BranchNode],
    config: SearchConfig,
    rng: np.random.Generator,
    branch_priors: Optional[dict] = None # Map branch_key -> float prob
) -> Optional[BranchNode]:
    """
    Select one BranchNode from candidates.
    
    If UCT mode: Random among unvisited, else Max UCT (using what stats?).
      Spec 13.1: "未訪問 Branch があれば、未訪問集合からランダムに1つ選ぶ"
      "そうでなければ UCT スコア最大を選ぶ" (Leaf->Branch edges have stats?)
      Spec 13 says: "Leaf→Branch 遷移もエッジ統計（N/W/inflight）を持つ。"
      BUT BranchNode ITSELF holds stats N/W. N(Branch) is effectively N(Leaf->Branch).
      So we can use BranchNode.N, BranchNode.W.
    
    If PUCT mode:
      Spec 13.2: "P_leaf(L,b) = softmax(V(b)/tau)" using Value head.
      "未訪問 Branch があれば、未訪問集合の中で P_leaf 最大を選ぶ"
      "そうでなければ、PUCT スコア最大を選ぶ"
    """
    if not branches:
        return None

    unvisited = [b for b in branches if b.N == 0] # Ignoring inflight for "unvisited" def? Or N+inflight==0?
    # Usually "unvisited" means N=0.
    
    if config.algorithm == "uct":
        if unvisited:
            return rng.choice(unvisited)
        
        # Max UCT
        # Score = Q(b) + c * sqrt(ln(N_parent) / N(b))
        # N_parent = leaf visit count? 
        # Actually Leaf doesn't track N explicitly in spec 5.1.
        # But sum(b.N) is total visits through this leaf.
        total_n = sum(b.N for b in branches)
        
        best_b = None
        best_score = -float('inf')
        
        for b in branches:
            q = b.W / max(1, b.N)
            u = config.c_uct * np.sqrt(np.log(total_n + 1) / (1 + b.N))
            vote = q + u
            if vote > best_score:
                best_score = vote
                best_b = b
        return best_b

    else: # PUCT
        # branch_priors required
        if branch_priors is None:
             # Fallback to uniform if not provided?
             # Or error.
             # Ideally simulation passes these.
             raise ValueError("branch_priors required for PUCT branch selection")

        if unvisited:
            # Pick unvisited with max Prior
            # Spec 13.2: "未訪問集合の中で P_leaf 最大を選ぶ"
            # Note: This is different from standard UCT random.
            best_b = max(unvisited, key=lambda b: branch_priors.get(b.branch_smiles, 0.0))
            return best_b
            
        # PUCT Score
        # score = Q_eff + c * P * sqrt(N_total) / (1 + N_eff)
        # N_total = sum(N)
        total_n = sum(b.N for b in branches)
        sqrt_n = np.sqrt(total_n + 1)
        
        best_b = None
        best_score = -float('inf')
        
        for b in branches:
            # Q_eff on BranchNode?
            # BranchNode holds stats.
            # Q_eff = (W - vloss * inflight) / N_eff
            # But BranchNode stats usually track downstream results FROM this branch.
            # Using them as Edge stats from Leaf->Branch is valid.
            
            # Note: BranchNode structure doesn't explicitly have inflight field for ITSELF?
            # Spec 5.3: "状態統計 N(s), W(s)". No inflight(s).
            # "行動統計... inflight(s,a)".
            # Spec 13: "Leaf→Branch 遷移もエッジ統計...を持つ"
            # Where?
            # "実装上は... BranchNode内... に集約" -> Refers to OUTGOING edges.
            # So Leaf->Branch edge stats are tricky.
            # Maybe we just use BranchNode state stats N, W?
            # And assume inflight is implied or we add it?
            # If we reuse Cached BranchNode, multiple leaves point to it (transposition).
            # So stats on BranchNode are aggregated from ALL parents.
            # Transposition is intended.
            # So yes, use BranchNode.N, BranchNode.W.
            
            # What about inflight for Branch?
            # BranchNode doesn't have 'inflight'.
            # Maybe we don't use vloss for Leaf->Branch selection?
            # Or we assume N is enough.
            
            q = b.W / max(1, b.N)
            prior = branch_priors.get(b.branch_smiles, 0.0)
            u = config.c_puct * prior * sqrt_n / (1 + b.N)
            vote = q + u
            
            if vote > best_score:
                best_score = vote
                best_b = b
                
        return best_b
