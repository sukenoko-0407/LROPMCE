import math
import numpy as np
from typing import List, Protocol
from rdkit import Chem
from rdkit.Chem import Descriptors
from ..config import RewardConfig

# Copied from example_code/reward_function.py
OPTIMAL_CLOGP = 2.5
DECAY_CONSTANT = -math.log(0.1) / ((OPTIMAL_CLOGP) ** 2)

def calculate_clogp(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return Descriptors.MolLogP(mol)
    except:
        return None

def clogp_reward(clogp: float) -> float:
    deviation = clogp - OPTIMAL_CLOGP
    reward = math.exp(-DECAY_CONSTANT * (deviation ** 2))
    return reward

# If we have other rewards, define them here.
# For now, only LogP is provided in example.

def aggregate_rewards(rewards: List[float], epsilon: float = 1e-12) -> float:
    """
    Geometric mean aggregation.
    R = exp( (1/N) * sum log(max(eps, r_i)) )
    """
    if not rewards:
        return 0.0
    
    log_sum = 0.0
    for r in rewards:
        # Constraint to [0,1] and epsilon
        r_clamped = max(epsilon, min(1.0, r))
        log_sum += math.log(r_clamped)
        
    avg_log = log_sum / len(rewards)
    return math.exp(avg_log)

class RewardCalculator:
    def __init__(self, config: RewardConfig):
        self.config = config
        
    def evaluate_smiles(self, smiles: str) -> float:
        # Currently hardcoded to use LogP reward only as per example availability
        # We can expand this list logic later
        
        # 1. Calc ClogP
        clogp = calculate_clogp(smiles)
        if clogp is None:
            return 0.0 # Error case
            
        r_clogp = clogp_reward(clogp)
        
        # 2. Aggregate (single element here)
        return aggregate_rewards([r_clogp], self.config.epsilon)

    def evaluate_batch(self, smiles_list: List[str]) -> List[float]:
        """
        Evaluate a list of SMILES strings.
        This provides a hook for optimization (e.g. vectorized calculation) later.
        For now, it iterates.
        """
        return [self.evaluate_smiles(s) for s in smiles_list]
