# Explainable Multi-Modal Fake News Detection
## Using Graph Neural Networks and Transformer Models on FakeNewsNet

**FINAL YEAR RESEARCH PROJECT — 2026**  
Siksha 'O' Anusandhan University | Faculty of Engineering & Technology (ITER)  
Department of Computer Science & Engineering  
**Section:** 2241031 | **Group No:** 31-11

---

## Team Members

| Name | Registration No |
|------|----------------|
| Ananya | 2241001077 |
| Vivek Kumar | 2241011211 |
| Anuj Mahato | 2241013033 |
| Lucky Pattanayak | 2241016336 |

---

## Project Overview

This project proposes an explainable multi-modal fake news detection framework that integrates Transformer-based NLP models (BERT/RoBERTa) with Graph Neural Networks (GCN/GAT) to detect fake news using both textual content and social propagation context from the **FakeNewsNet** dataset.

### Key Features
- **Multi-modal**: Combines text semantics + social network propagation
- **Transformer + GNN**: BERT/RoBERTa + GCN/GAT hybrid architecture
- **Cross-Attention Fusion**: Bi-directional attention between text and graph modalities
- **Explainable AI**: SHAP values, LIME token scores, attention heatmaps
- **High Performance**: ~92.4% accuracy, 0.919 F1-score on FakeNewsNet

---

## Architecture

```
News Text ──► BERT/RoBERTa Encoder ──► [CLS] Embedding (768-dim)
                                                    │
Social Graph ──► GAT Encoder ──► Graph Embedding (128-dim)
                                                    │
                        Cross-Attention Fusion (256-dim)
                                                    │
                         MLP Classifier Head
                                    │
                        ┌───────────┴───────────┐
                      REAL (0)               FAKE (1)
```

---

## Project Structure

```
fake_news_project/
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── README.md
├── src/
│   ├── config.py            # All hyperparameters & paths
│   ├── data_loader.py       # FakeNewsNet loading + graph construction
│   ├── text_model.py        # BERT/RoBERTa transformer encoder
│   ├── graph_model.py       # GCN / GAT encoder
│   ├── multimodal_fusion.py # Concat / Gated / Cross-Attention fusion
│   ├── classifier.py        # Full end-to-end model
│   ├── train.py             # Training loop + early stopping
│   ├── evaluate.py          # Metrics, confusion matrix, ROC curve
│   └── explainability.py    # SHAP, LIME, attention visualisation
├── notebooks/
│   └── FakeNewsDetection_Demo.ipynb  # Interactive demo
├── diagrams/
│   ├── 01_system_architecture.png
│   ├── 02_model_comparison.png
│   ├── 03_training_curves.png
│   ├── 04_confusion_matrix.png
│   ├── 05_roc_pr_curves.png
│   ├── 06_shap_importance.png
│   ├── 07_attention_heatmap.png
│   ├── 08_dataset_statistics.png
│   ├── 09_propagation_graph.png
│   └── 10_ablation_study.png
├── data/                    # Place FakeNewsNet CSVs here
├── models/                  # Saved model checkpoints
└── results/                 # Training history & evaluation results
```

---

## Installation

```bash
# 1. Clone and enter project
git clone <your-repo>
cd fake_news_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install PyTorch Geometric (required for GNN)
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

---

## Dataset Setup (FakeNewsNet)

Download the FakeNewsNet dataset:
```bash
git clone https://github.com/KaiDMML/FakeNewsNet
cd FakeNewsNet
pip install -r requirements.txt
python download_manager.py --dataset politifact --type fake --news_source all
python download_manager.py --dataset politifact --type real --news_source all
```

Place the resulting CSV files in `data/`:
- `data/politifact_real.csv`
- `data/politifact_fake.csv`
- `data/gossipcop_real.csv`
- `data/gossipcop_fake.csv`

> **Note:** If dataset files are not found, the pipeline automatically uses synthetic demo data for testing.

---

## Usage

### Train the model
```bash
python main.py --mode train
```

### Evaluate a saved model
```bash
python main.py --mode evaluate
```

### Explain a prediction
```bash
python main.py --mode explain --text "SHOCKING: Secret government documents leaked!"
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/FakeNewsDetection_Demo.ipynb
```

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| TF-IDF + SVM | 0.789 | 0.782 | 0.769 | 0.775 | 0.812 |
| LSTM | 0.821 | 0.815 | 0.808 | 0.811 | 0.857 |
| BERT (text only) | 0.873 | 0.869 | 0.865 | 0.867 | 0.921 |
| GCN (graph only) | 0.836 | 0.829 | 0.821 | 0.825 | 0.878 |
| BERT + GAT (gated) | 0.903 | 0.899 | 0.895 | 0.897 | 0.948 |
| **Ours: RoBERTa+GAT** | **0.924** | **0.921** | **0.917** | **0.919** | **0.963** |

---

## Configuration

All hyperparameters can be modified in `src/config.py`:

```python
BERT_MODEL_NAME = "roberta-base"   # or "bert-base-uncased"
GNN_TYPE        = "GAT"            # or "GCN"
FUSION_TYPE     = "attention"      # "concat" | "gated" | "attention"
NUM_EPOCHS      = 10
BATCH_SIZE      = 16
LEARNING_RATE   = 2e-5
```

---

## References

1. Shu, K., et al. "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information for Studying Fake News on Social Media." *Big Data*, 2020.
2. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers." *NAACL*, 2019.
3. Liu, Y., et al. "RoBERTa: A Robustly Optimized BERT Pretraining Approach." *arXiv*, 2019.
4. Velickovic, P., et al. "Graph Attention Networks." *ICLR*, 2018.
5. Bian, T., et al. "Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks." *AAAI*, 2020.
6. Lundberg, S., Lee, S. "A Unified Approach to Interpreting Model Predictions (SHAP)." *NeurIPS*, 2017.
7. Ribeiro, M.T., et al. "Why Should I Trust You? Explaining the Predictions of Any Classifier (LIME)." *KDD*, 2016.
8. Nguyen, V.H., et al. "Multi-modal fusion for fake news detection." *Information Processing & Management*, 2022.

---

## License

This project is for academic research purposes only. The FakeNewsNet dataset is subject to its original usage terms.
