"""SENTINEL — ESA spacecraft anomaly detection.

Public API is exposed via fully qualified imports, e.g.::

    from sentinel.params               import WINDOW_SIZE, MODELS_DIR
    from sentinel.ml_logic.data        import load_train, find_anomaly_segments
    from sentinel.ml_logic.scorer      import score_windows, score_report
    from sentinel.ml_logic.predictor   import predict_fe46, load_fe46_artefacts
    from sentinel.ml_logic.metrics     import event_f05, corrected_event_f05

Nothing is re-exported from this package's __init__ — that keeps imports
lazy (no eager matplotlib pull-in via viz, no TF pull-in via predictor)
and avoids drift between this list and the actual symbols in submodules.
"""
