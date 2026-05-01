"""
CLI orchestrator for SENTINEL preprocessing pipelines.

Usage:
    python -m sentinel.main preprocess          # 3-way split (data/processed/)
    python -m sentinel.main preprocess_kaggle   # Kaggle split (data/processed/kaggle/)
"""

import argparse


def preprocess() -> None:
    from .ml_logic.preprocessor import run_preprocessing
    run_preprocessing()


def preprocess_kaggle() -> None:
    from .ml_logic.preprocessor import run_preprocessing_kaggle
    run_preprocessing_kaggle()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel.main",
        description="SENTINEL preprocessing orchestrator",
    )
    parser.add_argument(
        "command",
        choices=["preprocess", "preprocess_kaggle"],
        help="preprocessing pipeline to run",
    )
    args = parser.parse_args()

    commands = {
        "preprocess":        preprocess,
        "preprocess_kaggle": preprocess_kaggle,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
