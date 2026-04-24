"""SENTINEL — ESA spacecraft anomaly detection."""
from .ml_logic.data import (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    load_train,
    load_test,
    load_target_channels,
    find_anomaly_segments,
    get_channel_cols,
    get_telecommand_cols,
)
from .ml_logic.metrics import f05_score, corrected_event_f05
from .ml_logic.preprocessor import (
    create_windows,
    run_preprocessing,
    run_preprocessing_kaggle,
)
from .ml_logic.viz import (
    plot_channels,
    plot_segment_zoom,
    plot_distributions,
    plot_correlation,
    plot_score_distribution,
    plot_score_timeline,
    plot_score_panels,
    plot_timeline,
    plot_event_zoom_with_score,
    plot_confusion_and_channel_errors,
)
