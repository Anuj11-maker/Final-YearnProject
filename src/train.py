"""
train.py
========
Advanced Training Pipeline for Explainable Multi-Modal Fake News Detection

Features:
- Mixed Precision Training (AMP)
- Warmup + Cosine LR Scheduler
- Early Stopping
- Gradient Clipping
- TensorBoard Logging
- Best Model Saving
- Focal Loss
- Stable Training History Return
- JSON Result Saving

Group 31-11 | SOA University | FRP-2026
"""

import os
import json
import time
import logging

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR
)

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)

from config import *
from data_loader import load_raw_csv, FakeNewsDataModule
from classifier import FakeNewsDetector

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, targets):

        ce_loss = self.ce(logits, targets)

        p_t = torch.exp(-ce_loss)

        loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss

        return loss.mean()


# ============================================================
# Metrics
# ============================================================

def compute_metrics(preds, labels, probs=None):

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0)
    }

    if probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(labels, probs)
            metrics["avg_precision"] = average_precision_score(labels, probs)
        except Exception:
            pass

    return metrics


# ============================================================
# Train One Epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler=None
):

    model.train()

    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # ====================================================
        # Mixed Precision Training
        # ====================================================

        if scaler is not None:

            with torch.amp.autocast(device_type="cuda"):

                logits, _, _ = model(
                    input_ids,
                    attention_mask
                )

                loss = criterion(logits, labels)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            scaler.step(optimizer)

            scaler.update()

        else:

            logits, _, _ = model(
                input_ids,
                attention_mask
            )

            loss = criterion(logits, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1]

        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.detach().cpu().numpy().tolist())

        all_labels.extend(labels.detach().cpu().numpy().tolist())

        all_probs.extend(probs.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)

    metrics = compute_metrics(
        all_preds,
        all_labels,
        all_probs
    )

    metrics["loss"] = avg_loss

    return metrics


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["label"].to(DEVICE)

        logits, _, _ = model(
            input_ids,
            attention_mask
        )

        loss = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1]

        preds = logits.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy().tolist())

        all_labels.extend(labels.cpu().numpy().tolist())

        all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)

    metrics = compute_metrics(
        all_preds,
        all_labels,
        all_probs
    )

    metrics["loss"] = avg_loss

    return metrics, all_preds, all_labels


# ============================================================
# Trainer
# ============================================================

class Trainer:

    def __init__(self, model, data_module):

        self.model = model.to(DEVICE)

        self.dm = data_module

        self.criterion = FocalLoss()

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                RESULTS_DIR,
                "tb_logs"
            )
        )

        self.best_val_f1 = 0.0

        self.patience = 0

        self.history = {
            "train": [],
            "val": []
        }

        # ====================================================
        # Optimizer
        # ====================================================

        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [

            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and
                    not any(nd in n for nd in no_decay)
                ],
                "weight_decay": WEIGHT_DECAY
            },

            {
                "params": [
                    p for n, p in model.named_parameters()
                    if p.requires_grad and
                    any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0
            }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=LEARNING_RATE
        )

        # ====================================================
        # Scheduler
        # ====================================================

        total_steps = NUM_EPOCHS * len(
            self.dm.train_loader()
        )

        warmup_steps = int(
            total_steps * WARMUP_RATIO
        )

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(1, warmup_steps)
        )

        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=1e-7
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler
            ],
            milestones=[warmup_steps]
        )

        # ====================================================
        # AMP Scaler
        # ====================================================

        self.scaler = (
            torch.amp.GradScaler("cuda")
            if DEVICE == "cuda"
            else None
        )

    # ========================================================
    # Training
    # ========================================================

    def fit(self):

        logger.info(
            f"[Trainer] Starting training on {DEVICE} "
            f"for {NUM_EPOCHS} epochs."
        )

        for epoch in range(1, NUM_EPOCHS + 1):

            start_time = time.time()

            train_metrics = train_one_epoch(
                self.model,
                self.dm.train_loader(),
                self.optimizer,
                self.criterion,
                self.scaler
            )

            val_metrics, _, _ = evaluate(
                self.model,
                self.dm.val_loader(),
                self.criterion
            )

            self.scheduler.step()

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
                f"Train Loss={train_metrics['loss']:.4f} "
                f"Acc={train_metrics['accuracy']:.4f} "
                f"F1={train_metrics['f1']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f} "
                f"Acc={val_metrics['accuracy']:.4f} "
                f"F1={val_metrics['f1']:.4f} | "
                f"Time={elapsed:.1f}s"
            )

            # =================================================
            # TensorBoard Logging
            # =================================================

            for k, v in train_metrics.items():
                self.writer.add_scalar(
                    f"Train/{k}",
                    v,
                    epoch
                )

            for k, v in val_metrics.items():
                self.writer.add_scalar(
                    f"Val/{k}",
                    v,
                    epoch
                )

            self.history["train"].append(train_metrics)

            self.history["val"].append(val_metrics)

            # =================================================
            # Save Best Model
            # =================================================

            if val_metrics["f1"] > self.best_val_f1:

                self.best_val_f1 = val_metrics["f1"]

                self.patience = 0

                if SAVE_BEST_MODEL:

                    save_path = os.path.join(
                        MODEL_SAVE_DIR,
                        "best_model.pt"
                    )

                    torch.save(
                        self.model.state_dict(),
                        save_path
                    )

                    logger.info(
                        f"→ Best model saved "
                        f"(Val F1={self.best_val_f1:.4f})"
                    )

            else:

                self.patience += 1

                if self.patience >= EARLY_STOPPING_PAT:

                    logger.info(
                        f"[Trainer] Early stopping "
                        f"at epoch {epoch}"
                    )

                    break

        self.writer.close()

        self.save_history()

        # IMPORTANT FIX
        return self.history

    # ========================================================
    # Save History
    # ========================================================

    def save_history(self):

        path = os.path.join(
            RESULTS_DIR,
            "training_history.json"
        )

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

        logger.info(
            f"[Trainer] Training history saved → {path}"
        )

    # ========================================================
    # Testing
    # ========================================================

    def test(self):

        best_path = os.path.join(
            MODEL_SAVE_DIR,
            "best_model.pt"
        )

        if os.path.exists(best_path):

            self.model.load_state_dict(
                torch.load(
                    best_path,
                    map_location=DEVICE
                )
            )

            logger.info(
                "[Trainer] Loaded best model for testing."
            )

        metrics, preds, labels = evaluate(
            self.model,
            self.dm.test_loader(),
            self.criterion
        )

        logger.info(
            f"[Test] "
            f"Acc={metrics['accuracy']:.4f} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f}"
        )

        # ====================================================
        # Save Results
        # ====================================================

        result_path = os.path.join(
            RESULTS_DIR,
            "test_results.json"
        )

        with open(result_path, "w") as f:

            json.dump({
                "metrics": metrics,
                "predictions": preds,
                "labels": labels
            }, f, indent=2)

        logger.info(
            f"[Trainer] Test results saved → {result_path}"
        )

        return metrics, preds, labels


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    from data_loader import set_seed

    set_seed()

    df = load_raw_csv(
        POLITIFACT_REAL_PATH,
        POLITIFACT_FAKE_PATH
    )

    dm = FakeNewsDataModule(df)

    model = FakeNewsDetector()

    trainer = Trainer(model, dm)

    history = trainer.fit()

    metrics, preds, labels = trainer.test()

    print("\n==============================")
    print(" FINAL TEST RESULTS ")
    print("==============================")

    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")