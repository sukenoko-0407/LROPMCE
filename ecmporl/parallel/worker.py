from ..config import SearchConfig, ConstraintConfig, RewardConfig
from ..fragments.library import FragmentLibrary
from ..model.inference import ModelInference
from ..mcts.engine import run_search_session

def run_worker(
    config: SearchConfig,
    constraint_config: ConstraintConfig,
    reward_config: RewardConfig,
    frag_lib: FragmentLibrary,
    initial_smiles: str,
    model_inference: ModelInference, 
    seed: int
):
    """
    Worker entry point.
    Runs a search session and returns the resulting tree.
    """
    return run_search_session(
        config,
        constraint_config,
        reward_config,
        frag_lib,
        initial_smiles,
        model_inference,
        seed
    )
