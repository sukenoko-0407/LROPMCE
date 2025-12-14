import torch
from torch_geometric.data import Batch
from typing import List, Dict, Tuple
import numpy as np
from ..model.pvnet import PolicyValueNet
from ..model.featurizer import get_graph_data

class ModelInference:
    def __init__(self, model: PolicyValueNet):
        self.model = model
        self.device = model.device
        self.model.eval()

    @torch.no_grad()
    def predict_priors_and_value(self, branch_smiles: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        # This is for BranchNode expansion/eval? 
        # Typically called "evaluate" in AlphaZero terms (returns P, v).
        
        if not branch_smiles:
            return [], []

        graphs = []
        valid_indices = []
        
        for i, smi in enumerate(branch_smiles):
            try:
                g = get_graph_data(smi)
                graphs.append(g)
                valid_indices.append(i)
            except Exception:
                pass
                
        if not graphs:
            return [None]*len(branch_smiles), [0.0]*len(branch_smiles)

        batch = Batch.from_data_list(graphs)
        # Forward
        policy_logits, values = self.model(batch)
        
        # To CPU
        policy_logits = policy_logits.cpu().numpy()
        values = values.cpu().numpy().flatten()
        
        final_policies = [None] * len(branch_smiles)
        final_values = [0.0] * len(branch_smiles)
        
        for i, idx in enumerate(valid_indices):
            final_policies[idx] = policy_logits[i]
            final_values[idx] = float(values[i])
            
        return final_policies, final_values

    @torch.no_grad()
    def predict_values(self, smiles_list: List[str]) -> List[float]:
        # For Leaf->Branch selection (value head only)
        # Returns list of values
        if not smiles_list:
            return []
            
        _, values = self.predict_priors_and_value(smiles_list)
        return values
