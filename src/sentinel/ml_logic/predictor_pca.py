from ..params import PCA_THRESHOLD
from .data import MODELS_DIR, PROCESSED_DIR
from .predictor import load_artefacts, predict as _predict

# Lazy cache so importing doesn't touch disk.
_pca      = None
_scaler   = None
_features = None


def _load():
    global _pca, _scaler, _features
    if _pca is None:
        _pca, _scaler, _features = load_artefacts(
            model_path  = MODELS_DIR / "pca.pkl",
            scaler_path = MODELS_DIR / "scaler.pkl",
            config_path = PROCESSED_DIR / "preprocessing_config.json",
        )


def predict(X_raw, threshold=PCA_THRESHOLD, return_scores=False):
    # Backwards-compatible: returns labels, or (labels, scores).
    _load()
    out = _predict(_pca, _scaler, _features, X_raw, threshold=threshold)
    if return_scores:
        return out["labels"], out["row_scores"]
    return out["labels"]


def predict_slice(df_raw, threshold=PCA_THRESHOLD):
    # Full dict for the NB 15 showcase plots.
    _load()
    return _predict(_pca, _scaler, _features, df_raw, threshold=threshold)
