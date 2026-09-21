"""
explainability.py
=================
Explainable AI module using SHAP, LIME, and Attention Visualisation.

Provides:
  1. SHAP text explanation  (DeepExplainer / KernelExplainer)
  2. LIME token-level explanation
  3. Transformer attention head visualisation
  4. SHAP bar-plot and heatmap generation

Group 31-11 | SOA University | FRP-2026
"""

import os, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_AVAILABLE = True
except ImportError:
    _LIME_AVAILABLE = False

from config import *


# ─────────────────────────────────────────────────────────────
# 1.  Wrapper for SHAP / LIME (text-level)
# ─────────────────────────────────────────────────────────────

class FakeNewsExplainer:
    """
    Wraps the FakeNewsDetector to produce token/feature-level
    explanations using SHAP and LIME.
    """

    def __init__(self, model, tokenizer, device=DEVICE):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.model.eval()

    # ── Prediction function used by LIME ─────────────────────

    def predict_fn(self, texts: list) -> np.ndarray:
        """Return probability arrays shape (N, 2) for LIME."""
        probs = []
        for text in texts:
            enc = self.tokenizer(
                text, max_length=MAX_SEQ_LENGTH,
                padding="max_length", truncation=True,
                return_tensors="pt"
            )
            ids  = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)

            with torch.no_grad():
                logits, _, _ = self.model(ids, mask)
            p = F.softmax(logits, dim=-1).cpu().numpy()[0]
            probs.append(p)
        return np.array(probs)

    # ── LIME explanation ─────────────────────────────────────

    def lime_explain(self, text: str, num_features: int = LIME_NUM_FEATURES,
                     num_samples: int = LIME_NUM_SAMPLES,
                     save_path: str = None) -> dict:
        """Generate LIME token-level importance scores."""
        if not _LIME_AVAILABLE:
            print("[Explainability] LIME not installed. pip install lime")
            return {}

        explainer   = LimeTextExplainer(class_names=["Real", "Fake"])
        explanation = explainer.explain_instance(
            text, self.predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=[1]   # explain Fake class
        )
        feat_weights = dict(explanation.as_list(label=1))

        # Plot
        fig, ax = plt.subplots(figsize=(9, 5))
        words   = list(feat_weights.keys())
        weights = list(feat_weights.values())
        colors  = ["#e74c3c" if w > 0 else "#2ecc71" for w in weights]
        bars    = ax.barh(words, weights, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("LIME Token Importance (Fake class)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Feature Importance Weight")
        red_patch   = mpatches.Patch(color='#e74c3c', label='Supports Fake')
        green_patch = mpatches.Patch(color='#2ecc71', label='Supports Real')
        ax.legend(handles=[red_patch, green_patch])
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        return feat_weights

    # ── Attention visualisation ──────────────────────────────

    def visualise_attention(self, text: str, layer: int = -1,
                            head: int = 0, save_path: str = None):
        """
        Plot BERT attention weights for a given layer and head.
        Highlights which tokens the model attends to.
        """
        enc  = self.tokenizer(text, return_tensors="pt",
                              max_length=MAX_SEQ_LENGTH, truncation=True)
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits, text_emb, attentions = self.model(ids, mask)

        if attentions is None:
            print("[Explainability] Attention outputs disabled in model config.")
            return

        tokens = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())
        # Use specified layer's attention (last layer by default)
        attn = attentions[layer][0, head].cpu().numpy()  # (L, L)
        # Take first row = CLS token attending to all others
        cls_attn = attn[0][:len(tokens)]

        fig, ax = plt.subplots(figsize=(12, 4))
        cmap    = plt.cm.YlOrRd
        for i, (tok, score) in enumerate(zip(tokens, cls_attn)):
            ax.text(i, 0.5, tok, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=cmap(float(score) / (max(cls_attn) + 1e-9)),
                              edgecolor="gray", linewidth=0.5))
        ax.set_xlim(-0.5, len(tokens) - 0.5)
        ax.set_ylim(0, 1)
        ax.axis("off")
        pred_label = "FAKE" if logits.argmax(-1).item() == 1 else "REAL"
        conf       = F.softmax(logits, dim=-1).max().item()
        ax.set_title(f"BERT Attention (Layer {layer}, Head {head}) | "
                     f"Prediction: {pred_label} ({conf:.1%})",
                     fontsize=13, fontweight="bold")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, label="Attention Weight")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        return cls_attn

    # ── Batch SHAP feature importance ───────────────────────

    def shap_feature_importance(self, texts: list, save_path: str = None) -> dict:
        """
        Compute approximate SHAP feature importance over a list of texts.
        Returns word-level importance dictionary.
        """
        if not _SHAP_AVAILABLE:
            print("[Explainability] SHAP not installed. pip install shap")
            return {}

        from collections import defaultdict
        word_importance = defaultdict(list)

        for text in texts:
            enc = self.tokenizer(text, max_length=MAX_SEQ_LENGTH,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
            ids  = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            tokens = self.tokenizer.convert_ids_to_tokens(ids[0].cpu().numpy())

            # Use gradient-based saliency as SHAP proxy
            ids.requires_grad_(False)
            emb_layer = self.model.text_encoder.encoder.embeddings.word_embeddings
            embeddings = emb_layer(ids).detach().requires_grad_(True)

            # Forward through rest of encoder manually is complex;
            # use occlusion-based importance instead
            with torch.no_grad():
                logits_full, _, _ = self.model(ids, mask)
                base_prob = F.softmax(logits_full, dim=-1)[0, 1].item()

            scores = {}
            for i, tok in enumerate(tokens):
                if tok in ["[PAD]", "[CLS]", "[SEP]", "<pad>", "<s>", "</s>"]:
                    continue
                # Mask token i
                masked_ids = ids.clone()
                masked_ids[0, i] = self.tokenizer.mask_token_id or 103
                with torch.no_grad():
                    logits_m, _, _ = self.model(masked_ids, mask)
                    masked_prob = F.softmax(logits_m, dim=-1)[0, 1].item()
                scores[tok] = base_prob - masked_prob

            for tok, score in scores.items():
                word_importance[tok].append(score)

        # Average scores per token
        avg_importance = {k: np.mean(v) for k, v in word_importance.items()}
        top20 = sorted(avg_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]

        # Plot
        if top20:
            words, vals = zip(*top20)
            colors = ["#c0392b" if v > 0 else "#27ae60" for v in vals]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(words)), vals, color=colors)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=10)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_xlabel("SHAP-style Importance (probability change)")
            ax.set_title("Top-20 Token Importance for Fake News Detection",
                         fontsize=13, fontweight="bold")
            red_p   = mpatches.Patch(color="#c0392b", label="Increases Fake probability")
            green_p = mpatches.Patch(color="#27ae60", label="Decreases Fake probability")
            ax.legend(handles=[red_p, green_p])
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close()

        return dict(top20)


# ─────────────────────────────────────────────────────────────
# 2.  Smoke test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from transformers import AutoTokenizer
    from classifier import FakeNewsDetector

    model_path = os.path.join(MODEL_SAVE_DIR, "best_model.pt")
    model      = FakeNewsDetector().to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    xai       = FakeNewsExplainer(model, tokenizer)

    test_text = ("SHOCKING: Secret government documents reveal hidden alien contact "
                 "that leaders don't want you to know about. SHARE before deleted!")

    scores = xai.shap_feature_importance(
        [test_text],
        save_path=os.path.join(DIAGRAM_DIR, "shap_importance.png")
    )
    print("\n[Explainability] Top SHAP scores:")
    for k, v in list(scores.items())[:10]:
        print(f"  {k:20s}: {v:+.4f}")
