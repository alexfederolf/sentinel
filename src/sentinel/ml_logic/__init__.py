"""Core building blocks for SENTINEL: data loading, metrics, preprocessing, visualisation."""
from .data import (
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
from .metrics import f05_score, corrected_event_f05
from .cv import (
    run_cv,
    run_sweep,
    load_cv_data,
    make_expanding_origin_folds,
    envr_cols,
    top_p_mean,
)
from .fusion import (
    event_diagnostics,
    fusion_diagnostics,
    format_fusion_report,
)
from .preprocessor import (
    create_windows,
    run_preprocessing,
    run_preprocessing_kaggle,
)
from .viz import (
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
