from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class SearchConfig:
    algorithm: Literal["uct", "puct"] = "puct"
    min_depth: int = 3
    max_depth: int = 12
    c_puct: float = 1.0
    c_uct: float = 1.0
    tau_policy: float = 1.0
    tau_branch: float = 1.0
    vloss: float = 1.0
    flush_threshold: int = 256
    max_simulations: int = 1000
    seed: int = 42
    value_target_mode: Literal["A", "A_prime"] = "A"
    num_search_workers: int = 1

@dataclass
class ConstraintConfig:
    HAC_min: Optional[int] = None
    HAC_max: Optional[int] = None
    MW_min: Optional[float] = None
    MW_max: Optional[float] = None
    hetero_max: Optional[int] = None
    chiral_max: Optional[int] = None

    def validate_leaf(self, props: dict) -> bool:
        """
        Check if leaf properties satisfy MINIMUM constraints (for ready checks).
        Maximum constraints are handled by legal action masking in BranchNode.
        """
        if self.HAC_min is not None and props.get("HAC", 0) < self.HAC_min:
            return False
        if self.MW_min is not None and props.get("MW", 0.0) < self.MW_min:
            return False
        return True

@dataclass
class FragmentLibraryConfig:
    frag_csv_path: str = "primary_elem_lib.csv"
    K_expected: Optional[int] = None

@dataclass
class ParallelConfig:
    max_workers: int = 4 # Default to safe small number
    enable_eval_cache: bool = False
    cache_backend: Literal["memory", "redis"] = "memory"
    cache_max_size: int = 100000

@dataclass
class RewardConfig:
    epsilon: float = 1e-12
    reward_timeout_sec: float = 5.0
    reward_parallelism: Literal["thread", "process"] = "thread"
    # Placeholder for list of reward function names or configs if needed
    reward_funcs: list[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    checkpoint_path: Optional[str] = None
    checkpoint_id: Optional[str] = None
    hidden_dim: int = 256
    num_layers: int = 3
    gnn_type: str = "AttentiveFP"
    embed_dim: int = 32
    num_timesteps: int = 2
    dropout: float = 0.1
    output_dim: int = 1
