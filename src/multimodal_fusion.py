"""
multimodal_fusion.py
====================
Multi-modal feature fusion strategies:
  1. Simple Concatenation
  2. Gated Fusion
  3. Cross-Attention Fusion  ← default (best accuracy)

Group 31-11 | SOA University | FRP-2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *


# ─────────────────────────────────────────────────────────────
# 1.  Concatenation Fusion (baseline)
# ─────────────────────────────────────────────────────────────

class ConcatFusion(nn.Module):
    """
    Simplest fusion: concatenate text and graph embeddings,
    then project to FUSION_DIM.
    """

    def __init__(self, text_dim: int = TRANSFORMER_HIDDEN,
                 graph_dim: int = GNN_OUT_CHANNELS,
                 out_dim:   int = FUSION_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + graph_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb  : (B, text_dim)
        graph_emb : (B, graph_dim)
        → fused   : (B, out_dim)
        """
        combined = torch.cat([text_emb, graph_emb], dim=-1)
        return self.proj(combined)


# ─────────────────────────────────────────────────────────────
# 2.  Gated Fusion
# ─────────────────────────────────────────────────────────────

class GatedFusion(nn.Module):
    """
    Learns a soft gate to dynamically weight text vs graph modalities.

    g = σ(W_g · [text; graph])
    fused = g ⊙ text_proj + (1-g) ⊙ graph_proj
    """

    def __init__(self, text_dim: int = TRANSFORMER_HIDDEN,
                 graph_dim: int = GNN_OUT_CHANNELS,
                 out_dim:   int = FUSION_DIM):
        super().__init__()
        self.text_proj  = nn.Linear(text_dim,  out_dim)
        self.graph_proj = nn.Linear(graph_dim, out_dim)
        self.gate       = nn.Linear(text_dim + graph_dim, out_dim)
        self.norm       = nn.LayerNorm(out_dim)
        self.drop       = nn.Dropout(0.3)
        self.out_dim    = out_dim

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        g = self.graph_proj(graph_emb)

        combined = torch.cat([text_emb, graph_emb], dim=-1)
        gate     = torch.sigmoid(self.gate(combined))

        fused = gate * t + (1 - gate) * g
        return self.drop(self.norm(fused))


# ─────────────────────────────────────────────────────────────
# 3.  Cross-Attention Fusion (best accuracy)
# ─────────────────────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Bi-directional cross-attention between text and graph streams.

    Text attends over graph tokens → enriched text
    Graph attends over text tokens → enriched graph
    Both enriched representations are concatenated and projected.
    """

    def __init__(self, text_dim: int = TRANSFORMER_HIDDEN,
                 graph_dim: int = GNN_OUT_CHANNELS,
                 out_dim:   int = FUSION_DIM,
                 num_heads: int = 8):
        super().__init__()

        # Project to common dimension
        self.d_model    = out_dim
        self.text_proj  = nn.Linear(text_dim,  self.d_model)
        self.graph_proj = nn.Linear(graph_dim, self.d_model)

        # Cross-attention modules
        self.text_cross_attn  = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=num_heads,
            dropout=0.1, batch_first=True)
        self.graph_cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=num_heads,
            dropout=0.1, batch_first=True)

        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model * 4, out_dim),
        )

        self.norm1  = nn.LayerNorm(self.d_model)
        self.norm2  = nn.LayerNorm(self.d_model)
        self.drop   = nn.Dropout(0.3)
        self.out_dim = out_dim

    def forward(self, text_emb: torch.Tensor, graph_emb: torch.Tensor) -> torch.Tensor:
        """
        text_emb  : (B, text_dim)
        graph_emb : (B, graph_dim)
        """
        # Project to d_model, add seq dimension
        t = self.text_proj(text_emb).unsqueeze(1)   # (B, 1, d_model)
        g = self.graph_proj(graph_emb).unsqueeze(1) # (B, 1, d_model)

        # Text queries graph
        t_enriched, _ = self.text_cross_attn(query=t, key=g, value=g)
        t_enriched    = self.norm1(t + t_enriched).squeeze(1)

        # Graph queries text
        g_enriched, _ = self.graph_cross_attn(query=g, key=t, value=t)
        g_enriched    = self.norm2(g + g_enriched).squeeze(1)

        # Combine and project
        combined = torch.cat([t_enriched, g_enriched], dim=-1)  # (B, 2*d_model)
        fused    = self.ff(combined)
        return self.drop(fused)


# ─────────────────────────────────────────────────────────────
# 4.  Factory
# ─────────────────────────────────────────────────────────────

def build_fusion(fusion_type: str = FUSION_TYPE) -> nn.Module:
    fusion_type = fusion_type.lower()
    if fusion_type == "concat":
        return ConcatFusion()
    elif fusion_type == "gated":
        return GatedFusion()
    elif fusion_type == "attention":
        return CrossAttentionFusion()
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                         "Choose 'concat', 'gated', or 'attention'.")


# ─────────────────────────────────────────────────────────────
# 5.  Unit test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B         = 4
    text_emb  = torch.randn(B, TRANSFORMER_HIDDEN)
    graph_emb = torch.randn(B, GNN_OUT_CHANNELS)

    for ft in ["concat", "gated", "attention"]:
        fusion = build_fusion(ft)
        out    = fusion(text_emb, graph_emb)
        print(f"[Fusion:{ft:10s}] output shape: {out.shape}")   # (4, 256)
