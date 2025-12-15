import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import numpy as np
from ..tree import MCTSTree
from ..config import SearchConfig

class ExperienceDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, np.ndarray, float]]):
        # sample: (smiles, policy_target, value_target)
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def extract_training_samples(
    tree: MCTSTree,
    config: SearchConfig,
    action_dim: int # K
) -> ExperienceDataset:
    """
    Extracts samples (s, pi, z).
    Mode 'A' (per-simulation) or 'A_prime' (per-node mean).
    We mostly implement A_prime logic here as "Avg Value at Node" is available in Tree.
    For 'A', we need simulation records which we didn't store fully yet (optional in tree).
    So let's implement A_prime: z = W(s)/N(s).
    """
    samples = []
    
    for b_node in tree.branches.values():
        if b_node.N < 1:
            continue
            
        # Policy Target
        # pi = N(s,a) / Sum_a N(s,a)
        # Only legal actions have entries?
        # We need full vector K? Yes usually.
        pi = np.zeros(action_dim, dtype=np.float32)
        
        sum_n = 0
        for act_id, stats in b_node.action_stats.items():
            pi[act_id] = stats.N
            sum_n += stats.N
            
        if sum_n > 0:
            pi /= sum_n
        else:
            # No visits yet? Skip
            continue
            
        # Value Target
        # z = W / N (Avg reward of sub-tree)
        # Note: W is accumulated reward backpropped through this node.
        z = b_node.W / b_node.N
        
        samples.append((b_node.branch_smiles, pi, z))
        
    return ExperienceDataset(samples)
