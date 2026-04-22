"""
CLI orchestrator for SENTINEL.

Single entry point for all pipeline stages. Each subcommand delegates
to the corresponding module in ml_logic/.

Usage:
    python -m sentinel.main preprocess         # default (three-way labelled split → data/processed/)
    python -m sentinel.main preprocess_kaggle  # Kaggle pipeline (80/20 + test set → data/processed/kaggle/)
    python -m sentinel.main pca_full           # fit PCA on all nominal windows + Kaggle submission
    python -m sentinel.main train              # train a model (not yet implemented)
    python -m sentinel.main predict            # generate submission (not yet implemented)
"""

import argparse


def preprocess() -> None:
    """Run the default three-way labelled preprocessing pipeline."""
    from .ml_logic.preprocessor import run_preprocessing
    run_preprocessing()


def preprocess_kaggle() -> None:
    """Run the Kaggle-submission preprocessing pipeline (80/20 + test set)."""
    from .ml_logic.preprocessor import run_preprocessing_kaggle
    run_preprocessing_kaggle()


def pca_full() -> None:
    """Fit PCA on all nominal windows and write the Kaggle submission (uses Kaggle pipeline arrays)."""
    from .ml_logic.pca_full import run_pca_full
    run_pca_full()


def train() -> None:
    """Train a model on the preprocessed arrays (placeholder)."""
    print("train — not implemented yet")


def predict() -> None:
    """Generate a Kaggle submission from a trained model (placeholder)."""
    print("predict — not implemented yet")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel.main",
        description="SENTINEL pipeline orchestrator",
    )
    parser.add_argument(
        "command",
        choices=["preprocess", "preprocess_kaggle", "pca_full", "train", "predict"],
        help="pipeline stage to run",
    )
    args = parser.parse_args()

    commands = {
        "preprocess":        preprocess,
        "preprocess_kaggle": preprocess_kaggle,
        "pca_full":          pca_full,
        "train":             train,
        "predict":           predict,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
