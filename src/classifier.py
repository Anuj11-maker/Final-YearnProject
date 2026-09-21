"""
classifier.py
=============
End-to-end FakeNewsDetector:
TransformerTextEncoder → GNN → Fusion → Classifier

Group 31-11 | SOA University | FRP-2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from text_model import TransformerTextEncoder
from graph_model import build_gnn, FallbackGNNEncoder
from multimodal_fusion import build_fusion

try:
    from torch_geometric.data import Data as GraphData, Batch
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


# ============================================================
# CLASSIFIER HEAD
# ============================================================

class ClassifierHead(nn.Module):

    def __init__(
        self,
        in_dim=FUSION_DIM,
        num_classes=NUM_CLASSES,
        dropout=CLASSIFIER_DROPOUT
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LayerNorm(in_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(in_dim * 2, in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),

            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


# ============================================================
# MAIN MODEL
# ============================================================

class FakeNewsDetector(nn.Module):

    def __init__(
        self,
        model_name=BERT_MODEL_NAME,
        gnn_type=GNN_TYPE,
        fusion_type=FUSION_TYPE
    ):
        super().__init__()

        # ----------------------------------------------------
        # TEXT ENCODER
        # ----------------------------------------------------
        self.text_encoder = TransformerTextEncoder(
            model_name=model_name
        )

        text_dim = self.text_encoder.hidden_size

        # ----------------------------------------------------
        # GRAPH ENCODER
        # ----------------------------------------------------
        if _PYG_AVAILABLE:
            self.gnn = build_gnn(gnn_type)
        else:
            self.gnn = FallbackGNNEncoder(
                in_dim=text_dim,
                out_dim=GNN_OUT_CHANNELS
            )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------
        self.fusion = build_fusion(fusion_type)

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------
        self.classifier = ClassifierHead(
            in_dim=self.fusion.out_dim
        )

    # ========================================================
    # CREATE SIMPLE GRAPH
    # ========================================================

    def build_simple_graph(self, text_emb):
        """
        Build simple chain graph from batch embeddings.
        """

        batch_size = text_emb.size(0)
        device = text_emb.device

        x = text_emb

        # simple chain edges
        edge_index = []

        for i in range(batch_size - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])

        if len(edge_index) == 0:
            edge_index = torch.tensor(
                [[0], [0]],
                dtype=torch.long,
                device=device
            )
        else:
            edge_index = torch.tensor(
                edge_index,
                dtype=torch.long,
                device=device
            ).t().contiguous()

        batch = torch.arange(
            batch_size,
            device=device
        )

        return x, edge_index, batch

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        input_ids,
        attention_mask,
        graph_batch=None
    ):

        # ----------------------------------------------------
        # TEXT FEATURES
        # ----------------------------------------------------
        text_emb, attentions = self.text_encoder(
            input_ids,
            attention_mask
        )

        # ----------------------------------------------------
        # GRAPH FEATURES
        # ----------------------------------------------------
        if _PYG_AVAILABLE:

            if graph_batch is not None:

                graph_emb = self.gnn(
                    graph_batch.x,
                    graph_batch.edge_index,
                    graph_batch.batch
                )

            else:
                # Build automatic graph
                x, edge_index, batch = self.build_simple_graph(
                    text_emb
                )

                graph_emb = self.gnn(
                    x,
                    edge_index,
                    batch
                )

        else:
            graph_emb = self.gnn(text_emb)

        # ----------------------------------------------------
        # FIX SHAPE
        # ----------------------------------------------------
        if graph_emb.size(0) != text_emb.size(0):

            if graph_emb.size(0) > text_emb.size(0):
                graph_emb = graph_emb[:text_emb.size(0)]

            else:
                repeat_factor = text_emb.size(0) // graph_emb.size(0) + 1
                graph_emb = graph_emb.repeat(repeat_factor, 1)
                graph_emb = graph_emb[:text_emb.size(0)]

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------
        fused = self.fusion(
            text_emb,
            graph_emb
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------
        logits = self.classifier(fused)

        return logits, text_emb, attentions

    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    def predict_proba(
        self,
        input_ids,
        attention_mask,
        graph_batch=None
    ):

        self.eval()

        with torch.no_grad():

            logits, _, _ = self.forward(
                input_ids,
                attention_mask,
                graph_batch
            )

            probs = F.softmax(logits, dim=-1)

        return probs

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    def count_parameters(self):

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        return {
            "total": total,
            "trainable": trainable
        }


# ============================================================
# UNIT TEST
# ============================================================

if __name__ == "__main__":

    model = FakeNewsDetector().to(DEVICE)

    params = model.count_parameters()

    print(f"Total params: {params['total']:,}")
    print(f"Trainable params: {params['trainable']:,}")

    B = 4
    L = 128

    ids = torch.randint(
        0,
        30522,
        (B, L)
    ).to(DEVICE)

    mask = torch.ones(
        B,
        L,
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        logits, text_emb, attn = model(
            ids,
            mask
        )

    print("Logits shape:", logits.shape)
    print("Text embedding shape:", text_emb.shape)