"""
config.py
=========
High Accuracy Configuration
Group 31-11 | FRP-2026
"""

import os
import torch

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DIAGRAM_DIR = os.path.join(BASE_DIR, "diagrams")

for folder in [DATA_DIR, MODEL_SAVE_DIR, RESULTS_DIR, DIAGRAM_DIR]:
    os.makedirs(folder, exist_ok=True)

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

DATASET_NAME = "FakeNewsNet"

POLITIFACT_REAL_PATH = os.path.join(DATA_DIR, "politifact_real.csv")
POLITIFACT_FAKE_PATH = os.path.join(DATA_DIR, "politifact_fake.csv")

GOSSIPCOP_REAL_PATH = os.path.join(DATA_DIR, "gossipcop_real.csv")
GOSSIPCOP_FAKE_PATH = os.path.join(DATA_DIR, "gossipcop_fake.csv")

# ─────────────────────────────────────────────
# Transformer Configuration
# ─────────────────────────────────────────────

# Higher accuracy transformer
BERT_MODEL_NAME = "roberta-base"

MAX_SEQ_LENGTH = 256

TRANSFORMER_HIDDEN = 768

# Freeze fewer layers for better learning
FREEZE_BERT_LAYERS = 1

# ─────────────────────────────────────────────
# GNN Configuration
# ─────────────────────────────────────────────

GNN_TYPE = "GAT"

GNN_IN_CHANNELS = 768
GNN_HIDDEN_CHANNELS = 256
GNN_OUT_CHANNELS = 256

GNN_NUM_LAYERS = 3
GNN_DROPOUT = 0.3

GAT_HEADS = 4

# ─────────────────────────────────────────────
# Fusion & Classifier
# ─────────────────────────────────────────────

FUSION_TYPE = "attention"

FUSION_DIM = 512

NUM_CLASSES = 2

CLASSIFIER_DROPOUT = 0.3

# ─────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────

SEED = 42

# Better stability
BATCH_SIZE = 16

# Better learning
LEARNING_RATE = 2e-5

# Prevent overfitting
WEIGHT_DECAY = 1e-4

# More training epochs
NUM_EPOCHS = 15

WARMUP_RATIO = 0.1

GRADIENT_CLIP = 1.0

SAVE_BEST_MODEL = True

# More patience before stopping
EARLY_STOPPING_PAT = 4

# Better dataset split
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ─────────────────────────────────────────────
# Explainability
# ─────────────────────────────────────────────

SHAP_BACKGROUND_SAMPLES = 50
SHAP_EXPLAIN_SAMPLES = 20

LIME_NUM_FEATURES = 10
LIME_NUM_SAMPLES = 300

ATTENTION_VISUALIZATION = True

# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Config] Using device: {DEVICE}")
print(f"[Config] Transformer : {BERT_MODEL_NAME}")
print(f"[Config] GNN type    : {GNN_TYPE}")