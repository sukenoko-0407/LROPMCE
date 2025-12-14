"""
PyTorch Geometric AttentiveFP Model for Molecular Property Prediction.

This module provides an AttentiveFP-based model compatible with the LeafBatchEvaluator protocol.
It uses the featurizer module to convert SMILES to graph data.
"""

import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP
from torch_geometric.data import Batch
from typing import Protocol, Optional

from featurizer import get_graph_data, get_atom_feat_dim, get_bond_feat_dim


class LeafBatchEvaluator(Protocol):
    """Protocol for batch evaluation of SMILES molecules."""
    def evaluate(self, leaf_smiles_list: list[str]) -> list[float]:
        ...


class AtomEmbedding(nn.Module):
    """
    Embedding layer for atom features.
    
    Converts index-encoded atom features into dense embeddings and concatenates them.
    """
    
    def __init__(self, embed_dim: int = 32):
        """
        Args:
            embed_dim: Embedding dimension for each atom feature.
        """
        super().__init__()
        self.embed_dim = embed_dim
        atom_feat_dims = get_atom_feat_dim()
        
        # Create embedding layers for each atom feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in atom_feat_dims
        ])
    
    @property
    def output_dim(self) -> int:
        """Total output dimension after concatenation."""
        return len(self.embeddings) * self.embed_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Atom features tensor of shape (num_atoms, num_features).
        
        Returns:
            Embedded atom features of shape (num_atoms, num_features * embed_dim).
        """
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x[:, i]))
        return torch.cat(embedded, dim=-1)


class BondEmbedding(nn.Module):
    """
    Embedding layer for bond features.
    
    Converts index-encoded bond features into dense embeddings and concatenates them.
    """
    
    def __init__(self, embed_dim: int = 32):
        """
        Args:
            embed_dim: Embedding dimension for each bond feature.
        """
        super().__init__()
        self.embed_dim = embed_dim
        bond_feat_dims = get_bond_feat_dim()
        
        # Create embedding layers for each bond feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in bond_feat_dims
        ])
    
    @property
    def output_dim(self) -> int:
        """Total output dimension after concatenation."""
        return len(self.embeddings) * self.embed_dim
    
    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: Bond features tensor of shape (num_edges, num_features).
        
        Returns:
            Embedded bond features of shape (num_edges, num_features * embed_dim).
        """
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(edge_attr[:, i]))
        return torch.cat(embedded, dim=-1)


class MolecularAttentiveFP(nn.Module):
    """
    AttentiveFP-based Graph Neural Network for molecular property prediction.
    
    Uses the AttentiveFP architecture from PyTorch Geometric with custom
    atom and bond embeddings based on the featurizer module.
    Compatible with the LeafBatchEvaluator protocol via the evaluate() method.
    """
    
    def __init__(
        self,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        device: Optional[str] = None
    ):
        """
        Args:
            embed_dim: Embedding dimension for each atom/bond feature category.
            hidden_dim: Hidden dimension for AttentiveFP layers.
            num_layers: Number of GNN layers in AttentiveFP.
            num_timesteps: Number of iterative refinement steps for global readout.
            dropout: Dropout rate.
            output_dim: Output dimension (1 for regression/binary classification).
            device: Device to run the model on ('cuda', 'cpu', or None for auto).
        """
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Embedding layers
        self.atom_embedding = AtomEmbedding(embed_dim)
        self.bond_embedding = BondEmbedding(embed_dim)
        
        # Calculate input dimensions
        in_channels = self.atom_embedding.output_dim  # 8 features * embed_dim
        edge_dim = self.bond_embedding.output_dim     # 4 features * embed_dim
        
        # AttentiveFP model
        self.attentive_fp = AttentiveFP(
            in_channels=in_channels,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            edge_dim=edge_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout
        )
    
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass for batched graph data.
        
        Args:
            batch: Batched graph data from PyTorch Geometric.
        
        Returns:
            Output predictions of shape (batch_size, output_dim).
        """
        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        edge_attr = batch.edge_attr.to(self.device)
        batch_idx = batch.batch.to(self.device)
        
        # Embed atom and bond features (convert discrete indices to continuous)
        x = self.atom_embedding(x)
        edge_attr = self.bond_embedding(edge_attr)
        
        # Forward through AttentiveFP
        out = self.attentive_fp(x, edge_index, edge_attr, batch_idx)
        
        return out
    
    def evaluate(self, leaf_smiles_list: list[str]) -> list[float]:
        """
        Evaluate a batch of SMILES strings.
        
        This method is compatible with the LeafBatchEvaluator protocol.
        
        Args:
            leaf_smiles_list: List of SMILES strings to evaluate.
        
        Returns:
            List of predicted values (floats) for each SMILES.
        """
        if not leaf_smiles_list:
            return []
        
        self.eval()
        
        # Convert SMILES to graph data
        graph_list = []
        valid_indices = []
        
        for i, smi in enumerate(leaf_smiles_list):
            try:
                graph = get_graph_data(smi)
                graph_list.append(graph)
                valid_indices.append(i)
            except ValueError:
                # Skip invalid SMILES, will return 0.0 for them
                pass
        
        # Initialize results with default values
        results = [0.0] * len(leaf_smiles_list)
        
        if not graph_list:
            return results
        
        # Batch the graphs
        batch = Batch.from_data_list(graph_list)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.forward(batch)
        
        # Convert to list of floats
        predictions = predictions.squeeze(-1).cpu().tolist()
        if isinstance(predictions, float):
            predictions = [predictions]
        
        # Assign predictions to valid indices
        for idx, pred in zip(valid_indices, predictions):
            results[idx] = pred
        
        return results
    
    def save(self, path: str) -> None:
        """Save model weights to file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model weights from file."""
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


def create_model(
    embed_dim: int = 32,
    hidden_dim: int = 256,
    num_layers: int = 3,
    num_timesteps: int = 2,
    dropout: float = 0.1,
    output_dim: int = 1,
    device: Optional[str] = None,
    weights_path: Optional[str] = None
) -> MolecularAttentiveFP:
    """
    Factory function to create a MolecularAttentiveFP model.
    
    Args:
        embed_dim: Embedding dimension for each atom/bond feature category.
        hidden_dim: Hidden dimension for AttentiveFP layers.
        num_layers: Number of GNN layers.
        num_timesteps: Number of iterative refinement steps for global readout.
        dropout: Dropout rate.
        output_dim: Output dimension.
        device: Device to run the model on.
        weights_path: Optional path to pre-trained weights.
    
    Returns:
        Configured MolecularAttentiveFP model.
    """
    model = MolecularAttentiveFP(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        output_dim=output_dim,
        device=device
    )
    
    if weights_path is not None:
        model.load(weights_path)
    
    model.to(model.device)
    
    return model


if __name__ == '__main__':
    # Example usage
    model = create_model(device='cpu')
    print(f"Model created on device: {model.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test with sample SMILES
    test_smiles = ['CCO', 'c1ccccc1', 'CC(=O)O', 'invalid_smiles']
    predictions = model.evaluate(test_smiles)
    
    for smi, pred in zip(test_smiles, predictions):
        print(f"SMILES: {smi:20s} -> Prediction: {pred:.4f}")
