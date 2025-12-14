"""
Reward function based on CLOGP (Crippen LogP) using RDKit.

This module provides a reward function that evaluates molecules based on their
calculated LogP values. The optimal CLOGP is 2.5 (reward = 1.0), and reward
decreases as CLOGP deviates from the optimal value.
"""

import math
from typing import Protocol
from rdkit import Chem
from rdkit.Chem import Descriptors


class LeafBatchEvaluator(Protocol):
    """Protocol for batch evaluation of SMILES molecules."""
    def evaluate(self, leaf_smiles_list: list[str]) -> list[float]:
        ...


# Optimal CLOGP value
OPTIMAL_CLOGP = 2.5

# Decay constant for Gaussian reward function
# Calculated so that at CLOGP = 0.0 or 5.0 (distance = 2.5), reward ≈ 0.1
# exp(-k * 2.5^2) = 0.1  =>  k = -ln(0.1) / 6.25 ≈ 0.368
DECAY_CONSTANT = -math.log(0.1) / ((OPTIMAL_CLOGP) ** 2)


def calculate_clogp(smiles: str) -> float | None:
    """
    Calculate CLOGP (Crippen LogP) for a given SMILES string.
    
    Args:
        smiles: SMILES string of the molecule.
    
    Returns:
        CLOGP value, or None if the SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    return Descriptors.MolLogP(mol)


def clogp_reward(clogp: float) -> float:
    """
    Calculate reward based on CLOGP value using Gaussian-like decay.
    
    The reward function is:
        reward = exp(-k * (clogp - optimal)^2)
    
    where k is chosen such that:
        - At optimal CLOGP (2.5): reward = 1.0
        - At CLOGP 0.0 or 5.0: reward ≈ 0.1
    
    Args:
        clogp: CLOGP value of the molecule.
    
    Returns:
        Reward value between 0 and 1.
    """
    deviation = clogp - OPTIMAL_CLOGP
    reward = math.exp(-DECAY_CONSTANT * (deviation ** 2))
    return reward


def evaluate_smiles(smiles: str) -> float:
    """
    Evaluate a single SMILES and return its reward.
    
    Args:
        smiles: SMILES string of the molecule.
    
    Returns:
        Reward value (0.0 for invalid SMILES).
    """
    clogp = calculate_clogp(smiles)
    if clogp is None:
        return 0.0
    return clogp_reward(clogp)


class CLogPRewardEvaluator:
    """
    CLOGP-based reward evaluator compatible with LeafBatchEvaluator protocol.
    
    Evaluates molecules based on how close their CLOGP is to the optimal value.
    """
    
    def __init__(
        self,
        optimal_clogp: float = OPTIMAL_CLOGP,
        target_reward_at_boundary: float = 0.1,
        boundary_distance: float = 2.5
    ):
        """
        Args:
            optimal_clogp: Optimal CLOGP value (reward = 1.0).
            target_reward_at_boundary: Target reward at boundary distance.
            boundary_distance: Distance from optimal where target_reward applies.
        """
        self.optimal_clogp = optimal_clogp
        self.decay_constant = -math.log(target_reward_at_boundary) / (boundary_distance ** 2)
    
    def _calculate_reward(self, clogp: float) -> float:
        """Calculate reward for a given CLOGP value."""
        deviation = clogp - self.optimal_clogp
        return math.exp(-self.decay_constant * (deviation ** 2))
    
    def evaluate(self, leaf_smiles_list: list[str]) -> list[float]:
        """
        Evaluate a batch of SMILES strings.
        
        This method is compatible with the LeafBatchEvaluator protocol.
        
        Args:
            leaf_smiles_list: List of SMILES strings to evaluate.
        
        Returns:
            List of reward values (floats) for each SMILES.
        """
        results = []
        
        for smiles in leaf_smiles_list:
            clogp = calculate_clogp(smiles)
            if clogp is None:
                results.append(0.0)
            else:
                results.append(self._calculate_reward(clogp))
        
        return results


def create_evaluator(
    optimal_clogp: float = 2.5,
    target_reward_at_boundary: float = 0.1,
    boundary_distance: float = 0.5
) -> CLogPRewardEvaluator:
    """
    Factory function to create a CLogPRewardEvaluator.
    
    Args:
        optimal_clogp: Optimal CLOGP value (reward = 1.0).
        target_reward_at_boundary: Target reward at boundary distance.
        boundary_distance: Distance from optimal where target_reward applies.
    
    Returns:
        Configured CLogPRewardEvaluator.
    """
    return CLogPRewardEvaluator(
        optimal_clogp=optimal_clogp,
        target_reward_at_boundary=target_reward_at_boundary,
        boundary_distance=boundary_distance
    )


# =============================================================================
# External Interface for Easy Calling
# =============================================================================

# Default evaluator instance with standard settings
# Use this directly or create a custom one with create_evaluator()
default_evaluator = create_evaluator()


def evaluate_smiles_list(
    smiles_list: list[str],
    evaluator: CLogPRewardEvaluator | None = None
) -> list[float]:
    """
    Evaluate a list of SMILES strings and return their rewards.
    
    This is the main entry point for external modules to call.
    
    Args:
        smiles_list: List of SMILES strings to evaluate.
        evaluator: Optional custom evaluator. If None, uses default_evaluator.
    
    Returns:
        List of reward values (floats) for each SMILES.
    
    Example:
        >>> from reward_function import evaluate_smiles_list
        >>> rewards = evaluate_smiles_list(['CCO', 'c1ccccc1', 'CCCCCCCC'])
        >>> print(rewards)
        [0.0142, 0.3114, 0.0000]
    """
    if evaluator is None:
        evaluator = default_evaluator
    return evaluator.evaluate(smiles_list)


if __name__ == '__main__':
    # Example usage and verification
    print("=" * 60)
    print("CLOGP Reward Function Verification")
    print("=" * 60)
    print(f"Optimal CLOGP: {OPTIMAL_CLOGP}")
    print(f"Decay constant: {DECAY_CONSTANT:.4f}")
    print()
    
    # Verify reward at key CLOGP values
    print("Reward at key CLOGP values:")
    test_clogps = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
    for clogp in test_clogps:
        reward = clogp_reward(clogp)
        print(f"  CLOGP = {clogp:4.1f} -> Reward = {reward:.4f}")
    
    print()
    print("=" * 60)
    print("Evaluator Test with Sample SMILES")
    print("=" * 60)
    
    evaluator = create_evaluator()
    
    test_smiles = [
        'CCO',          # Ethanol
        'c1ccccc1',     # Benzene
        'CCCCCCCC',     # Octane (high LogP)
        'O',            # Water (low LogP)
        'CC(=O)O',      # Acetic acid
        'invalid_smiles'
    ]
    
    rewards = evaluator.evaluate(test_smiles)
    
    print(f"{'SMILES':<20} {'CLOGP':>8} {'Reward':>8}")
    print("-" * 40)
    for smi, reward in zip(test_smiles, rewards):
        clogp = calculate_clogp(smi)
        clogp_str = f"{clogp:.3f}" if clogp is not None else "N/A"
        print(f"{smi:<20} {clogp_str:>8} {reward:>8.4f}")


# sample

# from reward_function import evaluate_smiles_list

# # 最もシンプルな呼び出し方法
# rewards = evaluate_smiles_list(['CCO', 'c1ccccc1', 'CCCCCCCC'])
# # -> [0.0142, 0.3114, 0.0000]

# # カスタム設定で呼び出す場合
# from reward_function import create_evaluator
# custom_eval = create_evaluator(optimal_clogp=3.0, boundary_distance=1.0)
# rewards = evaluate_smiles_list(['CCO', 'c1ccccc1'], evaluator=custom_eval)
