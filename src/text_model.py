"""
text_model.py
=============
Transformer encoder (BERT / RoBERTa / DistilBERT)
for semantic text representation.

Group 31-11 | SOA University | FRP-2026
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from config import *


class TransformerTextEncoder(nn.Module):
    """
    Wrapper for Transformer models.

    Supports:
    - BERT
    - RoBERTa
    - DistilBERT

    Features:
    - Optional layer freezing
    - CLS token extraction
    - Attention extraction for explainability
    """

    def __init__(
        self,
        model_name: str = BERT_MODEL_NAME,
        freeze_layers: int = FREEZE_BERT_LAYERS,
        output_attentions: bool = ATTENTION_VISUALIZATION,
    ):

        super().__init__()

        config = AutoConfig.from_pretrained(
            model_name,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        self.encoder = AutoModel.from_pretrained(
            model_name,
            config=config
        )

        self.hidden_size = self.encoder.config.hidden_size
        self.output_attentions = output_attentions

        # Freeze lower layers
        self._freeze_layers(freeze_layers)

        # Projection Head
        self.proj = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, TRANSFORMER_HIDDEN),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _freeze_layers(self, freeze_layers: int):
        """
        Freeze embeddings + lower transformer layers.

        Works for:
        - BERT
        - RoBERTa
        - DistilBERT
        """

        # ---------------------------------------------------------
        # Freeze embedding layer
        # ---------------------------------------------------------
        if hasattr(self.encoder, "embeddings"):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------
        # DistilBERT
        # Structure:
        # self.encoder.transformer.layer
        # ---------------------------------------------------------
        if hasattr(self.encoder, "transformer"):

            for i, layer in enumerate(self.encoder.transformer.layer):

                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        # ---------------------------------------------------------
        # BERT / RoBERTa
        # Structure:
        # self.encoder.encoder.layer
        # ---------------------------------------------------------
        elif hasattr(self.encoder, "encoder"):

            for i, layer in enumerate(self.encoder.encoder.layer):

                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        print(f"[TextEncoder] Frozen first {freeze_layers} transformer layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        """
        Parameters
        ----------
        input_ids      : (B, L)
        attention_mask : (B, L)

        Returns
        -------
        cls_emb     : (B, 768)
        attentions  : attention tensors
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token representation
        cls_emb = outputs.last_hidden_state[:, 0, :]

        # Projection
        cls_emb = self.proj(cls_emb)

        attentions = (
            outputs.attentions
            if self.output_attentions
            else None
        )

        return cls_emb, attentions


class TextFeatureExtractor(nn.Module):
    """
    Alternative wrapper using mean pooling.
    """

    def __init__(self, model_name: str = BERT_MODEL_NAME):
        super().__init__()

        self.encoder = TransformerTextEncoder(model_name)

    def mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:

        mask = attention_mask.unsqueeze(-1).float()

        summed = (last_hidden_state * mask).sum(dim=1)

        counts = mask.sum(dim=1).clamp(min=1e-9)

        return summed / counts

    def forward(self, input_ids, attention_mask):

        cls_emb, attentions = self.encoder(
            input_ids,
            attention_mask
        )

        return cls_emb, attentions


# ─────────────────────────────────────────────────────────────
# Unit Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    model = TransformerTextEncoder().to(DEVICE)

    B, L = 4, 128

    ids = torch.randint(0, 30522, (B, L)).to(DEVICE)

    mask = torch.ones(
        B,
        L,
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        emb, attn = model(ids, mask)

    print(f"[TextModel] CLS embedding shape : {emb.shape}")

    if attn:
        print(f"[TextModel] Attention layers : {len(attn)}")
        print(f"[TextModel] Attention[0] shape : {attn[0].shape}")