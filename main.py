"""
main.py
=======
Entry point for the Explainable Multi-Modal Fake News Detection project.

Group 31-11 | SOA University | FRP-2026
"""

import os
import sys
import argparse
import logging
import torch

# ------------------------------------------------------------
# Add src folder to path
# ------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from src.config import *

from src.data_loader import (
    load_raw_csv,
    FakeNewsDataModule,
    set_seed
)

from src.classifier import FakeNewsDetector

from src.train import Trainer

from src.evaluate import (
    ModelEvaluator,
    plot_model_comparison,
    plot_training_curves
)

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# TRAINING PIPELINE
# ============================================================

def run_training(args):

    logger.info("=" * 60)
    logger.info(" EXPLAINABLE MULTI-MODAL FAKE NEWS DETECTION ")
    logger.info(" Group 31-11 | SOA University | FRP-2026 ")
    logger.info("=" * 60)

    set_seed(SEED)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_raw_csv(
        POLITIFACT_REAL_PATH,
        POLITIFACT_FAKE_PATH
    )

    # --------------------------------------------------------
    # Data module
    # --------------------------------------------------------

    dm = FakeNewsDataModule(
        df,
        model_name=BERT_MODEL_NAME,
        batch_size=BATCH_SIZE
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = FakeNewsDetector(
        model_name=BERT_MODEL_NAME,
        gnn_type=GNN_TYPE,
        fusion_type=FUSION_TYPE,
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    params = model.count_parameters()

    logger.info(
        f"Model parameters → "
        f"Total: {params['total']:,} | "
        f"Trainable: {params['trainable']:,}"
    )

    # --------------------------------------------------------
    # Trainer
    # --------------------------------------------------------

    trainer = Trainer(model, dm)

    # IMPORTANT:
    # fit() MUST return self.history
    history = trainer.fit()

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    metrics, preds, labels = trainer.test()

    # ========================================================
    # PLOTS
    # ========================================================

    # --------------------------------------------------------
    # Training Curves
    # --------------------------------------------------------

    try:

        if history is not None:

            plot_training_curves(
                history,
                save_path=os.path.join(
                    DIAGRAM_DIR,
                    "03_training_curves.png"
                )
            )

            logger.info(
                "[OK] Training curves saved"
            )

        else:

            logger.warning(
                "[WARNING] Training history is None"
            )

    except Exception as e:

        logger.warning(
            f"[Plot Error] Training curves: {e}"
        )

    # --------------------------------------------------------
    # Model Comparison
    # --------------------------------------------------------

    try:

        plot_model_comparison(
            save_path=os.path.join(
                DIAGRAM_DIR,
                "02_model_comparison.png"
            )
        )

        logger.info(
            "[OK] Model comparison saved"
        )

    except Exception as e:

        logger.warning(
            f"[Plot Error] Model comparison: {e}"
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    evaluator = ModelEvaluator(
        model,
        dm.test_loader()
    )

    evaluator.run()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    try:

        evaluator.plot_confusion_matrix(
            save_path=os.path.join(
                DIAGRAM_DIR,
                "04_confusion_matrix.png"
            )
        )

        logger.info(
            "[OK] Confusion matrix saved"
        )

    except Exception as e:

        logger.warning(
            f"[Plot Error] Confusion matrix: {e}"
        )

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    try:

        evaluator.plot_roc_curve(
            save_path=os.path.join(
                DIAGRAM_DIR,
                "05_roc_pr_curves.png"
            )
        )

        logger.info(
            "[OK] ROC curve saved"
        )

    except Exception as e:

        logger.warning(
            f"[Plot Error] ROC curve: {e}"
        )

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    try:

        evaluator.save_results()

        logger.info(
            "[OK] Evaluation report saved"
        )

    except Exception as e:

        logger.warning(
            f"[Save Error] Evaluation results: {e}"
        )

    logger.info(
        "\n✓ Training complete. "
        "All outputs saved to results/ and diagrams/"
    )

    return history, metrics


# ============================================================
# EVALUATION MODE
# ============================================================

def run_evaluate(args):

    set_seed(SEED)

    df = load_raw_csv(
        POLITIFACT_REAL_PATH,
        POLITIFACT_FAKE_PATH
    )

    dm = FakeNewsDataModule(df)

    model = FakeNewsDetector()

    model_path = os.path.join(
        MODEL_SAVE_DIR,
        "best_model.pt"
    )

    if os.path.exists(model_path):

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=DEVICE
            )
        )

        logger.info(
            f"[Evaluate] Loaded model from {model_path}"
        )

    else:

        logger.warning(
            "[Evaluate] No saved model found"
        )

    evaluator = ModelEvaluator(
        model,
        dm.test_loader()
    )

    evaluator.run()

    evaluator.plot_confusion_matrix(
        save_path=os.path.join(
            DIAGRAM_DIR,
            "04_confusion_matrix.png"
        )
    )

    evaluator.plot_roc_curve(
        save_path=os.path.join(
            DIAGRAM_DIR,
            "05_roc_pr_curves.png"
        )
    )

    evaluator.save_results()

    return evaluator.metrics()


# ============================================================
# EXPLAINABILITY MODE
# ============================================================

def run_explain(args):

    from transformers import AutoTokenizer
    from src.explainability import FakeNewsExplainer

    set_seed(SEED)

    model = FakeNewsDetector()

    model_path = os.path.join(
        MODEL_SAVE_DIR,
        "best_model.pt"
    )

    if os.path.exists(model_path):

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=DEVICE
            )
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_MODEL_NAME
    )

    explainer = FakeNewsExplainer(
        model,
        tokenizer
    )

    text = args.text or (
        "Breaking news: secret government alien contact exposed."
    )

    logger.info(f"[Explain] {text}")

    # --------------------------------------------------------
    # Attention Heatmap
    # --------------------------------------------------------

    explainer.visualise_attention(
        text,
        save_path=os.path.join(
            DIAGRAM_DIR,
            "07_attention_heatmap.png"
        )
    )

    # --------------------------------------------------------
    # SHAP Importance
    # --------------------------------------------------------

    explainer.shap_feature_importance(
        [text],
        save_path=os.path.join(
            DIAGRAM_DIR,
            "06_shap_importance.png"
        )
    )

    logger.info(
        "[OK] Explainability diagrams generated"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "explain"],
        default="train"
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Modes
    # --------------------------------------------------------

    if args.mode == "train":

        run_training(args)

    elif args.mode == "evaluate":

        run_evaluate(args)

    elif args.mode == "explain":

        run_explain(args)


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()