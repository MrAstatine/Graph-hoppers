"""
gcn_encoder.py
--------------
GCN-based graph encoder for Paradigm A (GNN -> DRL).

The encoder takes raw Cora node features + edge_index and produces
fixed-size node embeddings. These embeddings are later pooled and fed
into the SB3 PPO policy as the observation vector.

Architecture:
  Input (1433-dim Cora features)
  -> GCNConv (hidden_dim)  + ReLU + Dropout
  -> GCNConv (hidden_dim)  + ReLU + Dropout
  -> GCNConv (embed_dim)
  -> mean-pool over all nodes  -> (embed_dim,) vector  [used as RL obs]

The same encoder is also used as the standalone GCN baseline by adding
a linear classification head on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(nn.Module):
    """
    Two-layer GCN that maps node features to node embeddings.
    Graph-level representation = mean of all node embeddings.
    """

    def __init__(
        self,
        in_channels: int = 1433,   # Cora raw feature dim
        hidden_dim: int = 64,
        embed_dim: int = 32,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, embed_dim)
        self.dropout = dropout
        self.embed_dim = embed_dim

    def forward(self, x, edge_index):
        """
        Args:
            x:          [N, in_channels] node feature matrix
            edge_index: [2, E] COO edge index

        Returns:
            node_emb:   [N, embed_dim] per-node embeddings
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index)
        return h  # [N, embed_dim]

    def graph_embedding(self, x, edge_index):
        """Mean-pool node embeddings -> single graph-level vector."""
        node_emb = self.forward(x, edge_index)
        return node_emb.mean(dim=0)  # [embed_dim]


class GCNClassifier(nn.Module):
    """
    Full GCN baseline for node classification.
    Encoder + linear head. Used in Phase II baseline training.
    """

    def __init__(
        self,
        in_channels: int = 1433,
        hidden_dim: int = 64,
        embed_dim: int = 32,
        num_classes: int = 7,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, embed_dim, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index):
        """
        Returns:
            logits: [N, num_classes]
        """
        node_emb = self.encoder(x, edge_index)
        return self.classifier(node_emb)
