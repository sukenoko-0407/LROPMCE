import numpy as np
from ..config import ConstraintConfig
from .library import FragmentLibrary

def get_legal_actions(
    current_props: dict,
    lib: FragmentLibrary,
    config: ConstraintConfig
) -> np.ndarray:
    """
    Returns array of legal action IDs (int).
    Filters based on MAX constraints.
    """
    legal_mask = np.ones(lib.K, dtype=bool)

    # HAC max
    if config.HAC_max is not None:
        remaining = config.HAC_max - current_props.get("HAC", 0)
        legal_mask &= (lib.delta_hac <= remaining)

    # MW max
    if config.MW_max is not None:
        remaining = config.MW_max - current_props.get("MW", 0.0)
        legal_mask &= (lib.delta_mw <= remaining)

    # Hetero max
    if config.hetero_max is not None:
        remaining = config.hetero_max - current_props.get("cnt_hetero", 0)
        legal_mask &= (lib.delta_hetero <= remaining)

    # Chiral max
    if config.chiral_max is not None:
        remaining = config.chiral_max - current_props.get("cnt_chiral", 0)
        legal_mask &= (lib.delta_chiral <= remaining)

    return np.where(legal_mask)[0]
