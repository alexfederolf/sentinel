"""Tests for sentinel.ml_logic.submission."""
import numpy as np
import pandas as pd
import pytest

from kaggle.submission import (
    default_submission_path,
    make_submission,
    submission_summary,
)


def test_make_submission_writes_and_returns_dataframe(tmp_path):
    ids   = np.arange(100, dtype=np.int64)
    preds = np.zeros(100, dtype=np.int64)
    preds[10:20] = 1
    out = tmp_path / "sub.parquet"

    df = make_submission(preds, ids, out_path=out, validate_schema=False)
    assert out.exists()
    assert list(df.columns) == ["id", "is_anomaly"]
    assert df["is_anomaly"].dtype == np.uint8
    # Round-trip through parquet
    df_r = pd.read_parquet(out)
    assert (df_r["id"].values == ids).all()
    assert (df_r["is_anomaly"].values == preds.astype(np.uint8)).all()


def test_make_submission_rejects_non_binary(tmp_path):
    ids   = np.arange(5, dtype=np.int64)
    preds = np.array([0, 1, 2, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="binary"):
        make_submission(preds, ids, tmp_path / "bad.parquet",
                        validate_schema=False)


def test_make_submission_length_mismatch(tmp_path):
    ids   = np.arange(10, dtype=np.int64)
    preds = np.zeros(9, dtype=np.int64)
    with pytest.raises(ValueError, match="length mismatch"):
        make_submission(preds, ids, tmp_path / "bad.parquet",
                        validate_schema=False)


def test_make_submission_validates_against_sample(tmp_path):
    ids   = np.arange(50, dtype=np.int64)
    preds = np.zeros(50, dtype=np.int64)
    sample_path = tmp_path / "sample_submission.parquet"
    pd.DataFrame({"id": ids, "is_anomaly": np.zeros(50, np.uint8)}) \
        .to_parquet(sample_path, index=False)

    df = make_submission(
        preds, ids,
        out_path=tmp_path / "out.parquet",
        sample_submission_path=sample_path,
        validate_schema=True,
    )
    assert len(df) == 50


def test_make_submission_sample_id_mismatch_raises(tmp_path):
    ids   = np.arange(50, dtype=np.int64)
    preds = np.zeros(50, dtype=np.int64)
    # Sample has a different id order → must raise
    sample_path = tmp_path / "sample_submission.parquet"
    shuffled = ids.copy()[::-1]
    pd.DataFrame({"id": shuffled, "is_anomaly": np.zeros(50, np.uint8)}) \
        .to_parquet(sample_path, index=False)

    with pytest.raises(ValueError, match="id order"):
        make_submission(
            preds, ids,
            out_path=tmp_path / "out.parquet",
            sample_submission_path=sample_path,
            validate_schema=True,
        )


def test_make_submission_missing_sample_raises(tmp_path):
    ids   = np.arange(5, dtype=np.int64)
    preds = np.zeros(5, dtype=np.int64)
    with pytest.raises(FileNotFoundError):
        make_submission(
            preds, ids,
            out_path=tmp_path / "out.parquet",
            sample_submission_path=tmp_path / "does_not_exist.parquet",
            validate_schema=True,
        )


def test_submission_summary_counts():
    y = np.zeros(100, dtype=np.uint8)
    y[10:20] = 1     # segment A, length 10
    y[50:53] = 1     # segment B, length 3
    df = pd.DataFrame({"id": np.arange(100), "is_anomaly": y})
    s = submission_summary(df)
    assert s["rows"] == 100
    assert s["positive_rows"] == 13
    assert s["positive_rate"] == pytest.approx(0.13)
    assert s["n_segments"] == 2


def test_submission_summary_all_zero():
    df = pd.DataFrame({"id": np.arange(10),
                       "is_anomaly": np.zeros(10, dtype=np.uint8)})
    s = submission_summary(df)
    assert s["positive_rows"] == 0
    assert s["n_segments"]    == 0
    assert s["positive_rate"] == 0.0


def test_default_submission_path_uses_submissions_dir():
    p = default_submission_path("my_model")
    assert p.name == "my_model.parquet"
    assert p.parent.name == "submissions"
