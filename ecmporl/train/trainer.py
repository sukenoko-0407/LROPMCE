import torch
from torch.utils.data import DataLoader
from typing import Optional
from ..model.pvnet import PolicyValueNet
from ..model.featurizer import get_graph_data
from torch_geometric.data import Batch

def train_policy_value_model(
    model: PolicyValueNet,
    dataset,
    optimizer,
    epochs: int = 1,
    batch_size: int = 32
):
    model.train()
    criterion_pol = torch.nn.CrossEntropyLoss() # or KLDiv
    criterion_val = torch.nn.MSELoss()
    
    # Custom collate for PyG
    def collate_fn(batch_list):
        smiles_list, pis, zs = zip(*batch_list)
        
        # Helper to convert smiles to graphs
        graphs = []
        valid_mask = []
        for i, smi in enumerate(smiles_list):
            try:
                g = get_graph_data(smi)
                graphs.append(g)
                valid_mask.append(True)
            except:
                valid_mask.append(False)
        
        if not graphs:
             return None, None, None
             
        batch_graph = Batch.from_data_list(graphs)
        
        # Filter targets
        pis_valid = [p for p, v in zip(pis, valid_mask) if v]
        zs_valid = [z for z, v in zip(zs, valid_mask) if v]
        
        pis_tensor = torch.tensor(np.stack(pis_valid), dtype=torch.float32)
        zs_tensor = torch.tensor(np.stack(zs_valid), dtype=torch.float32)
        
        return batch_graph, pis_tensor, zs_tensor

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    import numpy as np # needed inside
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for batch_g, batch_pi, batch_z in loader:
            if batch_g is None:
                continue
                
            batch_g = batch_g.to(model.device)
            batch_pi = batch_pi.to(model.device)
            batch_z = batch_z.to(model.device)
            
            optimizer.zero_grad()
            
            pred_pi, pred_v = model(batch_g)
            pred_v = pred_v.squeeze(-1)
            
            # Loss
            # Policy: CrossEntropy expects class indices or probs?
            # If target is probs (pi), use CrossEntropy with soft targets or KLDiv.
            # PyTorch CrossEntropyLoss supports prob targets in newer versions.
            # Assuming recent torch.
            loss_p = criterion_pol(pred_pi, batch_pi)
            loss_v = criterion_val(pred_v, batch_z)
            
            loss = loss_p + loss_v
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
    return model
