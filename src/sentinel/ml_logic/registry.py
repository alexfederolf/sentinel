"""
Model registry — save and load trained models.

Single responsibility: knows WHERE models live on disk and
how to serialize / deserialize them. Nothing else.

Functions
---------
save_model(model, name)   — pickle a model to models/
load_model(name)          — load a pickled model from models/
load_scaler()             — load the fitted RobustScaler
"""

import pickle
from pathlib import Path

# from .data import MODELS_DIR
from ..params import MODELS_DIR


def save_model(model, name: str = "model") -> Path:
    """
    Save a model to models/<name>.pkl

    Parameters
    ----------
    model : any picklable object (sklearn, PCA, etc.)
    name  : filename without extension

    Returns
    -------
    Path where the model was saved
    """
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved → {path}")
    return path


def load_model(name: str = "pca") -> object:
    """
    Load a pickled model from models/<name>.pkl

    Parameters
    ----------
    name : filename without extension (default: pca_bootcamp)

    Returns
    -------
    The loaded model object

    Raises
    ------
    FileNotFoundError if the model file doesn't exist
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No model found at {path}. "
            f"Run 'python -m sentinel.main preprocess' first."
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded ← {path}")
    return model


def load_scaler() -> object:
    """
    Load the fitted RobustScaler from models/scaler.pkl

    Returns
    -------
    The fitted RobustScaler object

    Raises
    ------
    FileNotFoundError if the scaler file doesn't exist
    """
    path = MODELS_DIR / "scaler.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No scaler found at {path}. "
            f"Run 'python -m sentinel.main preprocess' first."
        )
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler loaded ← {path}")
    return scaler
