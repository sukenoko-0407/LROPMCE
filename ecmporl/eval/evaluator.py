from typing import List, Protocol
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .reward import RewardCalculator
from ..config import RewardConfig
import time

class LeafBatchEvaluator(Protocol):
    def evaluate(self, leaf_smiles_list: List[str]) -> List[float]:
        ...

class SyncEvaluator:
    def __init__(self, calc: RewardCalculator):
        self.calc = calc
        
    def evaluate(self, leaf_smiles_list: List[str]) -> List[float]:
        # Use batch eval if available
        if hasattr(self.calc, "evaluate_batch"):
             return self.calc.evaluate_batch(leaf_smiles_list)

        results = []
        for smi in leaf_smiles_list:
            try:
                val = self.calc.evaluate_smiles(smi)
                results.append(val)
            except Exception:
                results.append(0.0) # Fail safe
        return results

class ParallelEvaluator:
    def __init__(self, calc: RewardCalculator, max_workers: int = 4, mode: str = "thread"):
        self.calc = calc
        self.max_workers = max_workers
        self.mode = mode
        # For process pool, 'calc' must be picklable.
        
    def evaluate(self, leaf_smiles_list: List[str]) -> List[float]:
        # Simple parallel map
        # Note: If mode is process, ensure global functions are used or picklable objects.
        # Here we use a helper method if needed.
        
        if self.mode == "process":
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor

        with Executor(max_workers=self.max_workers) as executor:
            # We map self.calc.evaluate_smiles. 
            # Pickling valid? RewardCalculator stores config (dataclass), clean enough.
            results = list(executor.map(self.calc.evaluate_smiles, leaf_smiles_list))
            
        return results
