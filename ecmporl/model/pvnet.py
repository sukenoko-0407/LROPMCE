import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Batch, Data
from typing import Optional, List
from .featurizer import get_atom_feat_dim, get_bond_feat_dim, get_graph_data

class AtomEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        atom_feat_dims = get_atom_feat_dim()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in atom_feat_dims
        ])
    
    @property
    def output_dim(self) -> int:
        return len(self.embeddings) * self.embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x[:, i]))
        return torch.cat(embedded, dim=-1)

class BondEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        bond_feat_dims = get_bond_feat_dim()
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in bond_feat_dims
        ])
    
    @property
    def output_dim(self) -> int:
        return len(self.embeddings) * self.embed_dim
    
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(edge_attr[:, i]))
        return torch.cat(embedded, dim=-1)

class PolicyValueNet(nn.Module):
    def __init__(
        self,
        action_dim: int, # K
        embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None
    ):
        super().__init__()
        
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.action_dim = action_dim
        
        self.atom_embedding = AtomEmbedding(embed_dim)
        self.bond_embedding = BondEmbedding(embed_dim)
        
        in_channels = self.atom_embedding.output_dim
        edge_dim = self.bond_embedding.output_dim
        
        # Using AttentiveFP as backbone.
        # But AttentiveFP in PyG usually gives 1 output or graph embedding?
        # Standard AttentiveFP class forward returns 'out' which is graph-level embedding 
        # IF out_channels is set? Or node embeddings?
        # PyG AttentiveFP doc: forward(x, edge_index, edge_attr, batch) -> (N, out_channels) or (BATCH, out_channels)
        # depending on implementation. 
        # Wait, the example code uses AttentiveFP directly with out_channels=1.
        # We want to extract features first, then heads.
        # But PyG AttentiveFP couples them.
        # Alternative: Use AttentiveFP to get graph feature, then split heads.
        # IF we set out_channels to hidden_dim, maybe we get a vector?
        
        self.backbone = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim, # Get feature vector
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
        
        # Heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim) # Logits
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh() # Value in [-1, 1] or [0, 1]? 
            # Spec says "value: shape (1,) - [0,1]のスカラー"
            # So Sigmoid or Tanh scaled?
            # Usually AlphaGoZero uses Tanh [-1, 1].
            # But requirements say [0, 1] (or rather, reward is [0,1]).
            # Let's use Sigmoid if reward is [0,1].
            # "Value target z ... R = exp(...)" -> [0, 1].
        )
        self.value_activation = nn.Sigmoid()

    def forward(self, batch: Batch):
        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        edge_attr = batch.edge_attr.to(self.device)
        batch_idx = batch.batch.to(self.device)
        
        x_emb = self.atom_embedding(x)
        edge_emb = self.bond_embedding(edge_attr)
        
        # Backbone
        h = self.backbone(x_emb, edge_index, edge_emb, batch_idx)
        
        # Heads
        policy_logits = self.policy_head(h)
        value_logits = self.value_head(h)
        value = self.value_activation(value_logits)
        
        return policy_logits, value

    def batch_value(self, smiles_list: List[str]) -> torch.Tensor:
        """Helper for Leaf->Branch value estimation (batch inference)"""
        # ... implementation later or use inference module
        pass
