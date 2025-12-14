import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os

class FragmentLibrary:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Fragment library not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        required_cols = ["id", "element", "mw", "hac", "cnt_hetero", "cnt_chiral"]
        # Allow slight variations if needed, but strict based on example
        if not all(col in df.columns for col in required_cols):
             raise ValueError(f"CSV must contain columns: {required_cols}")

        self.K = len(df)
        
        # Ensure IDs are 0..K-1 contiguous
        ids = df["id"].sort_values().values
        if not np.array_equal(ids, np.arange(self.K)):
            raise ValueError("Fragment IDs must be contiguous from 0 to K-1")

        self.df = df.set_index("id").sort_index()
        
        # Prepare arrays for fast access
        self.smiles = self.df["element"].values
        # Delta props
        self.delta_hac = self.df["hac"].values
        self.delta_mw = self.df["mw"].values
        self.delta_hetero = self.df["cnt_hetero"].values
        self.delta_chiral = self.df["cnt_chiral"].values

    def get_fragment_smiles(self, action_id: int) -> str:
        return self.smiles[action_id]

    def get_deltas(self, action_id: int) -> Dict[str, Any]:
        return {
            "HAC": self.delta_hac[action_id],
            "MW": self.delta_mw[action_id],
            "cnt_hetero": self.delta_hetero[action_id],
            "cnt_chiral": self.delta_chiral[action_id]
        }
