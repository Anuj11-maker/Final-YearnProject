"""
graph_model.py
==============
Graph Neural Network (GCN / GAT) for propagation-pattern modelling.

The news article and its social-interaction users form a heterogeneous
graph. The GNN learns to capture how misinformation spreads.

Group 31-11 | SOA University | FRP-2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports – graceful fallback if PyG not installed
try:
    from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data as GraphData, Batch
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False

from config import *


# ─────────────────────────────────────────────────────────────
# 1.  GCN Encoder
# ─────────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """
    Multi-layer Graph Convolutional Network.

    Architecture:
        GCNConv(in → hidden) → BN → ReLU → Dropout
        GCNConv(hidden → hidden) × (num_layers - 2)
        GCNConv(hidden → out) → GlobalMeanPool → out_emb
    """

    def __init__(self,
                 in_channels:     int = GNN_IN_CHANNELS,
                 hidden_channels: int = GNN_HIDDEN_CHANNELS,
                 out_channels:    int = GNN_OUT_CHANNELS,
                 num_layers:      int = GNN_NUM_LAYERS,
                 dropout:         float = GNN_DROPOUT):
        super().__init__()
        assert _PYG_AVAILABLE, "torch_geometric is required for GCNEncoder."

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.bns.append(nn.BatchNorm1d(out_channels))

        self.dropout = dropout
        self.out_channels = out_channels

    def forward(self, x, edge_index, batch):
        """
        x          : (N_total, in_channels)
        edge_index : (2, E_total)
        batch      : (N_total,)  – batch assignment vector

        Returns
        -------
        graph_emb : (B, out_channels)
        """
        for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = F.relu(x)

        # Global mean pooling → one vector per graph
        graph_emb = global_mean_pool(x, batch)
        return graph_emb


# ─────────────────────────────────────────────────────────────
# 2.  GAT Encoder (Graph Attention Network)
# ─────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Multi-head Graph Attention Network encoder.

    Uses multi-head attention to weight neighbour contributions,
    capturing which users are influential in spreading fake news.
    """

    def __init__(self,
                 in_channels:     int = GNN_IN_CHANNELS,
                 hidden_channels: int = GNN_HIDDEN_CHANNELS,
                 out_channels:    int = GNN_OUT_CHANNELS,
                 heads:           int = GAT_HEADS,
                 dropout:         float = GNN_DROPOUT):
        super().__init__()
        assert _PYG_AVAILABLE, "torch_geometric is required for GATEncoder."

        self.conv1 = GATConv(in_channels,     hidden_channels,
                             heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=heads, dropout=dropout, concat=True)
        self.conv3 = GATConv(hidden_channels * heads, out_channels,
                             heads=1, dropout=dropout, concat=False)

        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.dropout      = dropout
        self.out_channels = out_channels

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn1(self.conv1(x, edge_index)))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn2(self.conv2(x, edge_index)))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn3(self.conv3(x, edge_index)))

        graph_emb = global_mean_pool(x, batch)
        return graph_emb


# ─────────────────────────────────────────────────────────────
# 3.  Factory
# ─────────────────────────────────────────────────────────────

def build_gnn(gnn_type: str = GNN_TYPE) -> nn.Module:
    """Return either a GCNEncoder or GATEncoder."""
    if gnn_type.upper() == "GCN":
        return GCNEncoder()
    elif gnn_type.upper() == "GAT":
        return GATEncoder()
    else:
        raise ValueError(f"Unknown GNN type: {gnn_type}. Choose 'GCN' or 'GAT'.")


# ─────────────────────────────────────────────────────────────
# 4.  Fallback MLP encoder (when PyG unavailable)
# ─────────────────────────────────────────────────────────────

class FallbackGNNEncoder(nn.Module):
    """
    Simple MLP that processes news-node features directly
    when torch_geometric is not installed.
    """

    def __init__(self, in_dim=GNN_IN_CHANNELS, out_dim=GNN_OUT_CHANNELS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, GNN_HIDDEN_CHANNELS),
            nn.BatchNorm1d(GNN_HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(GNN_DROPOUT),
            nn.Linear(GNN_HIDDEN_CHANNELS, out_dim),
            nn.ReLU(),
        )
        self.out_channels = out_dim

    def forward(self, x, *args, **kwargs):
        return self.mlp(x)


# ─────────────────────────────────────────────────────────────
# 5.  Unit test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if _PYG_AVAILABLE:
        B   = 4   # graphs in batch
        gnn = build_gnn("GAT").to(DEVICE)

        # Simulate a batched graph
        from torch_geometric.data import Batch
        graphs = []
        for _ in range(B):
            n = 10    # nodes per graph
            x = torch.randn(n, GNN_IN_CHANNELS)
            edges = torch.randint(0, n, (2, 20))
            graphs.append(GraphData(x=x, edge_index=edges))

        batch = Batch.from_data_list(graphs).to(DEVICE)
        out   = gnn(batch.x, batch.edge_index, batch.batch)
        print(f"[GNN] Output shape: {out.shape}")   # (4, 128)
    else:
        print("[GNN] PyG not available; using FallbackGNNEncoder.")
        enc = FallbackGNNEncoder().to(DEVICE)
        x   = torch.randn(4, GNN_IN_CHANNELS).to(DEVICE)
        print(f"[GNN-Fallback] Output: {enc(x).shape}")
