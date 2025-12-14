import numpy as np
import math
from typing import Optional, Tuple
from ..nodes import BranchNode
from ..config import SearchConfig

def select_action(
    branch: BranchNode,
    config: SearchConfig,
    rng: np.random.Generator
) -> Tuple[int, bool]:
    """
    Select an action from valid moves.
    Returns (action_id, is_newly_expanded).
    
    If there are unexpanded legal actions, pick one (randomly or by prior).
    If all expanded, use UCT/PUCT score.
    """
    legal_indices = branch.legal_actions
    if len(legal_indices) == 0:
        # Should be terminal?
        raise ValueError("No legal actions for branch")

    # Check for unexpanded actions
    # An action is unexpanded if it's not in action_stats or N=0 (and inflight=0?)
    # Spec 12.2.2: "Q_eff=0.0 (unexpanded)"
    # UCT spec 12.1: "Unexpanded (unvisited) action if any -> random 1"
    
    # Let's clean up "unexpanded" definition.
    # We maintain action_stats only for visited/inflight actions.
    # So any legal action NOT in action_stats is definitely unexpanded.
    
    expanded_set = set(branch.action_stats.keys())
    unexpanded = [a for a in legal_indices if a not in expanded_set]

    if unexpanded:
        # Pick one.
        # PUCT: "prior P(s,a) が高い未展開行動が優先的に選択される" (12.2.2)?
        # Text says: "未展開行動（...）の Q_eff は 0.0 とする", "Prior P(s,a) が高い...優先的に"
        # Wait, if Q=0, score = 0 + c * P * sqrt(N) / 1.
        # So yes, highest P will be picked.
        # BUT, standard UCT picks random unexpanded.
        
        if config.algorithm == "puct":
            # For PUCT, we usually compute scores for ALL, treating unexpanded as having Q=0, N=0.
            # So we fall through to score calculation.
            pass
        else:
            # UCT: Randomly pick one unexpanded
            action = rng.choice(unexpanded)
            return int(action), True

    # Compute scores for ALL legal actions (both expanded and unexpanded for PUCT)
    best_score = -float('inf')
    best_action = -1
    
    # Precompute N_total
    # PUCT: N_total = sum_{a in legal} N(s,a) (without inflight)
    # UCT: N_total = sum_{a in legal} N(s,a) (or N_eff for vloss?)
    # Spec 10.2: "N_total_eff(s) = sum_{a in legal} N_eff(s,a)" if vloss used.
    
    # Let's assume vloss is used (recommended).
    vloss = config.vloss
    
    total_n_eff = 0.0
    for a in legal_indices:
        stats = branch.action_stats.get(a)
        if stats:
            total_n_eff += stats.N + stats.inflight
    
    # For PUCT 12.2.2: N_total is "confirmed visit only" -> sum N(s,a)
    total_n_confirmed = sum(branch.action_stats[a].N for a in legal_indices if a in branch.action_stats)
    
    sqrt_n = 0.0
    if config.algorithm == "puct":
        sqrt_n = math.sqrt(total_n_confirmed + 1)
    else:
        # UCT uses log
        # score = Q + c * sqrt(ln(N_total)/N)
        # N_total here usually is parent visits? Or sum of siblings.
        # Spec 10.2: sqrt( ln(N_total_eff + 1) / (1 + N_eff) )
        sqrt_n = math.sqrt(math.log(total_n_eff + 1)) if total_n_eff > 0 else 0.0

    # Iterate all legal
    # Need to handle legal_indices mapping to priors if PUCT
    
    for i, action in enumerate(legal_indices):
        stats = branch.action_stats.get(action)
        
        # Stats
        n_eff = 0
        q_eff = 0.0
        
        if stats:
            n_eff = stats.N + stats.inflight
            # Check pending child exclusion
            if stats.child_leaf:
                 # We need tree access to check child.leaf_calc == "pending"
                 # BUT select_action might not have tree access easily?
                 # passed 'branch' node only.
                 # Spec says: "Already expanded action pointing to pending child -> exclude"
                 # But we don't have the child node here.
                 # Actually, 'inflight > 0' usually implies pending or evaluating.
                 # If inflight > 0, do we exclude?
                 # No, vloss handles it.
                 # Explicit exclusion: "Selection の結果「候補が全て pending で空」になった場合は flush要求"
                 # "Already expanded action pointing to child Leaf that is 'pending' ... exclude"
                 # We can use the 'inflight' count as a proxy? 
                 # If inflight > 0, it is pending? Yes, mostly.
                 # Requirement 10.3: "BranchNode の Action 選択において、既に展開済みの Action が指す子 Leaf が leaf_calc=='pending' の場合、その Action は候補から除外"
                 # If we assume inflight==1 implies pending...
                 # But wait, vloss is designed to ALLOW parallel exploration of same branch if score is high enough?
                 # "pending Leaf の Selection 除外" seems conflicting with "vloss".
                 # If we return a pending leaf, we just add to waiters (10.1).
                 # So we DO NOT exclude it, we just don't want to expand it AGAIN if it's already pending?
                 # Ah, "Leaf が pending" means we are waiting for value.
                 # If we pick it again, we just add to waiters.
                 # The "Selection 除外" requirement in 10.3 says "exclude (score = -inf)".
                 # Why? Maybe to force exploration of other branches while waiting?
                 # If we exclude it, we effectively force Width search.
                 # Okay, if `inflight > 0`, treat as excluded?
                 # Or strictly check child node status?
                 # Let's check `stats.inflight > 0`. If inflight, logic says we are waiting.
                 # If explicit requirement 10.3 says exclude, we exclude.
                 if stats.inflight > 0:
                     continue # Exclude actions that are currently "in flight"
            
            if n_eff > 0:
                q_eff = (stats.W - vloss * stats.inflight) / n_eff
        
        score = 0.0
        
        if config.algorithm == "puct":
            # P(s,a)
            prior = branch.priors_legal[i] if branch.priors_legal is not None else (1.0 / len(legal_indices))
            
            u_score = config.c_puct * prior * sqrt_n / (1 + n_eff)
            score = q_eff + u_score
            
        else:
            # UCT
            # If UCT and unexpanded, we shouldn't be here (handled above)
            # Unless we are in the "all expanded" block.
            # But wait, logic above says: if unexpanded and UCT -> random return.
            # So if we are here in UCT, all are expanded.
            
            u_score = config.c_uct * sqrt_n / math.sqrt(1 + n_eff)
            score = q_eff + u_score

        if score > best_score:
            best_score = score
            best_action = action
        elif score == best_score:
             # Tie-break: min action_id
             if action < best_action or best_action == -1:
                 best_action = action
                 
    if best_action == -1:
        # Could happen if all filtered out?
        return -1, False
        
    # Check if newly expanded (not in stats)
    is_new = (best_action not in branch.action_stats)
    return int(best_action), is_new
