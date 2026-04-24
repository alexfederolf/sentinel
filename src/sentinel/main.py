"""
CLI orchestrator for SENTINEL

JUST PLACEHOLDER

"""

import argparse


def preprocess() -> None:
    from .ml_logic.preprocessor import run_preprocessing
    run_preprocessing()

def train() -> None:
    print("train — not implemented yet")

def predict() -> None:
    print("predict — not implemented yet")

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sentinel.main",
        description="SENTINEL pipeline orchestrator",
    )
    parser.add_argument(
        "command",
        choices=["preprocess", "train", "predict"],
        help="pipeline stage to run",
    )
    args = parser.parse_args()

    commands = {
        "preprocess":        preprocess,
        "train":             train,
        "predict":           predict,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
