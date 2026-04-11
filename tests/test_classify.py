"""
Phase 5 — Classification tests.

Synthetic tabular data is generated via sklearn.datasets.make_classification.
Raster-based tests (extract_training_samples, predict_raster) use the
make_raster fixture from conftest.py plus rasterio for label raster creation.
No external data files required.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from sklearn.datasets import make_classification

from pipeline.classify import (
    ClassificationConfig,
    ClassificationResult,
    _build_model,
    _extract_feature_importances,
    _minority_f1,
    apply_quality_gate,
    apply_smote,
    detect_imbalance,
    extract_training_samples,
    predict_raster,
    train_model,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_data() -> tuple[np.ndarray, np.ndarray]:
    """Well-separated binary data; any decent classifier achieves > 90% OA."""
    X, y = make_classification(
        n_samples=300, n_features=4, n_classes=2,
        n_informative=4, n_redundant=0, class_sep=5.0,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture
def imbalanced_data() -> tuple[np.ndarray, np.ndarray]:
    """95 / 5 class split — minority class < default 10% threshold."""
    X, y = make_classification(
        n_samples=300, n_features=4, n_classes=2,
        weights=[0.95, 0.05], n_informative=2, n_redundant=0,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.int32)


@pytest.fixture
def noise_data() -> tuple[np.ndarray, np.ndarray]:
    """Random labels — expected OA ≈ 50%; will fail a 0.95 OA gate."""
    rng = np.random.default_rng(0)
    X   = rng.random((200, 4)).astype(np.float32)
    y   = rng.integers(0, 2, 200).astype(np.int32)
    return X, y


@pytest.fixture
def feature_names_4() -> list[str]:
    return ["feat_a", "feat_b", "feat_c", "feat_d"]


def _make_label_raster(
    path: Path,
    width: int,
    height: int,
    data: np.ndarray,   # (height, width) int32
    nodata: int = 0,
    crs_epsg: int = 32633,
    res: float = 10.0,
) -> Path:
    """Write a single-band integer label raster to path."""
    transform = from_origin(500_000.0, 5_000_200.0, res, res)
    profile   = {
        "driver":    "GTiff",
        "dtype":     "int32",
        "width":     width,
        "height":    height,
        "count":     1,
        "crs":       CRS.from_epsg(crs_epsg),
        "transform": transform,
        "nodata":    nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data[np.newaxis])
    return path


# ── detect_imbalance ──────────────────────────────────────────────────────────

def test_detect_imbalance_balanced_returns_false(balanced_data):
    X, y = balanced_data
    is_imb, _ = detect_imbalance(y, threshold_pct=10.0)
    assert not is_imb


def test_detect_imbalance_imbalanced_returns_true(imbalanced_data):
    X, y = imbalanced_data
    is_imb, _ = detect_imbalance(y, threshold_pct=10.0)
    assert is_imb


def test_detect_imbalance_pct_sums_to_100():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    _, class_pct = detect_imbalance(y, threshold_pct=10.0)
    assert pytest.approx(sum(class_pct.values()), rel=1e-6) == 100.0


def test_detect_imbalance_pct_keys_match_classes():
    y = np.array([1, 1, 2, 2, 3], dtype=np.int32)
    _, class_pct = detect_imbalance(y, threshold_pct=5.0)
    assert set(class_pct.keys()) == {1, 2, 3}


def test_detect_imbalance_pct_formula():
    """pct_of_total = class_count / total_samples * 100 (not relative to majority)."""
    y          = np.array([0] * 90 + [1] * 10, dtype=np.int32)
    _, pct     = detect_imbalance(y, threshold_pct=5.0)
    assert pct[0] == pytest.approx(90.0)
    assert pct[1] == pytest.approx(10.0)


# ── apply_smote ───────────────────────────────────────────────────────────────

def test_apply_smote_output_size_increases(imbalanced_data):
    X, y = imbalanced_data
    X_r, y_r = apply_smote(X, y, random_state=0)
    assert len(y_r) > len(y)


def test_apply_smote_feature_count_preserved(imbalanced_data):
    X, y      = imbalanced_data
    X_r, y_r  = apply_smote(X, y)
    assert X_r.shape[1] == X.shape[1]


def test_apply_smote_minority_class_count_increases(imbalanced_data):
    X, y      = imbalanced_data
    from collections import Counter
    before    = Counter(y.tolist())[1]
    _, y_r    = apply_smote(X, y)
    after     = Counter(y_r.tolist())[1]
    assert after > before


def test_apply_smote_output_is_c_contiguous(imbalanced_data):
    X, y     = imbalanced_data
    X_r, y_r = apply_smote(X, y)
    assert X_r.flags["C_CONTIGUOUS"]
    assert y_r.flags["C_CONTIGUOUS"]


# ── extract_training_samples ──────────────────────────────────────────────────

def test_extract_training_samples_correct_shapes(make_raster, tmp_path):
    feat_path  = make_raster("feat.tif", bands=4, width=32, height=32,
                             dtype="float32", nodata=-9999.0)
    # Label raster: class 1 and class 2; nodata_label=0 already absent since
    # we fill with alternating 1/2
    label_data = np.ones((32, 32), dtype=np.int32)
    label_data[16:, :] = 2
    lab_path   = _make_label_raster(tmp_path / "labels.tif", 32, 32, label_data)

    X, y = extract_training_samples(feat_path, lab_path)
    assert X.ndim  == 2
    assert y.ndim  == 1
    assert X.shape == (len(y), 4)
    assert len(y)  == 32 * 32   # all pixels labeled


def test_extract_training_samples_nodata_excluded(make_raster, tmp_path):
    feat_path  = make_raster("feat_nd.tif", bands=2, width=16, height=16,
                             dtype="float32", nodata=-9999.0)
    # All pixels labeled 1 except top-left 4×4 block → nodata_label=0
    label_data = np.ones((16, 16), dtype=np.int32)
    label_data[:4, :4] = 0   # unlabeled
    lab_path   = _make_label_raster(tmp_path / "labels_nd.tif", 16, 16, label_data)

    X, y = extract_training_samples(feat_path, lab_path, nodata_label=0)
    assert len(y) == 16 * 16 - 4 * 4   # 240 labeled pixels


def test_extract_training_samples_max_samples_respected(make_raster, tmp_path):
    feat_path  = make_raster("feat_max.tif", bands=4, width=64, height=64,
                             dtype="float32")
    label_data = np.ones((64, 64), dtype=np.int32)
    label_data[32:, :] = 2
    lab_path   = _make_label_raster(tmp_path / "labels_max.tif", 64, 64, label_data)

    X, y = extract_training_samples(feat_path, lab_path, max_samples=100)
    assert len(y) <= 100


def test_extract_training_samples_stratified_proportions(make_raster, tmp_path):
    """Stratified subsampling must preserve class proportions approximately."""
    feat_path  = make_raster("feat_strat.tif", bands=4, width=64, height=64,
                             dtype="float32")
    # 50/50 class split
    label_data            = np.ones((64, 64), dtype=np.int32)
    label_data[32:, :]    = 2
    lab_path              = _make_label_raster(
        tmp_path / "labels_strat.tif", 64, 64, label_data
    )

    X, y = extract_training_samples(feat_path, lab_path, max_samples=200,
                                    random_state=0)
    from collections import Counter
    counts = Counter(y.tolist())
    # Each class should be ~50% of 200 = 100 ± 5 (rounding budget)
    assert abs(counts[1] - counts[2]) <= 10


def test_extract_training_samples_shape_mismatch_raises(make_raster, tmp_path):
    feat_path  = make_raster("feat_mm.tif", bands=2, width=16, height=16)
    label_data = np.ones((32, 32), dtype=np.int32)
    lab_path   = _make_label_raster(tmp_path / "labels_mm.tif", 32, 32, label_data)
    with pytest.raises(ValueError, match="shape mismatch"):
        extract_training_samples(feat_path, lab_path)


def test_extract_training_samples_no_labeled_raises(make_raster, tmp_path):
    feat_path  = make_raster("feat_nl.tif", bands=2, width=16, height=16)
    label_data = np.zeros((16, 16), dtype=np.int32)   # all nodata_label=0
    lab_path   = _make_label_raster(tmp_path / "labels_nl.tif", 16, 16, label_data)
    with pytest.raises(ValueError, match="No labeled"):
        extract_training_samples(feat_path, lab_path, nodata_label=0)


# ── apply_quality_gate ────────────────────────────────────────────────────────

def test_gate_passes_when_both_metrics_meet_thresholds():
    passed, msg = apply_quality_gate(
        oa=0.85, minority_f1=0.75,
        cfg={"min_oa_threshold": 0.80, "min_minority_f1": 0.70},
    )
    assert passed
    assert "passed" in msg.lower()


def test_gate_fails_when_oa_below_threshold():
    passed, msg = apply_quality_gate(
        oa=0.70, minority_f1=0.80,
        cfg={"min_oa_threshold": 0.80, "min_minority_f1": 0.70},
    )
    assert not passed
    assert "OA" in msg


def test_gate_fails_when_minority_f1_below_threshold():
    passed, msg = apply_quality_gate(
        oa=0.90, minority_f1=0.60,
        cfg={"min_oa_threshold": 0.80, "min_minority_f1": 0.70},
    )
    assert not passed
    assert "minority F1" in msg


def test_gate_fails_when_both_metrics_below_threshold():
    passed, msg = apply_quality_gate(
        oa=0.50, minority_f1=0.40,
        cfg={"min_oa_threshold": 0.80, "min_minority_f1": 0.70},
    )
    assert not passed
    assert "OA" in msg and "minority F1" in msg


def test_gate_at_exact_threshold_passes():
    """Gate uses >= so exact threshold values must pass."""
    passed, _ = apply_quality_gate(
        oa=0.80, minority_f1=0.70,
        cfg={"min_oa_threshold": 0.80, "min_minority_f1": 0.70},
    )
    assert passed


# ── train_model — Random Forest ───────────────────────────────────────────────

def test_train_model_rf_result_fields(balanced_data, feature_names_4):
    X, y  = balanced_data
    cfg   = {"min_oa_threshold": 0.50, "min_minority_f1": 0.50,
             "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg  = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                  k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)

    assert result.model_type == "random_forest"
    assert set(result.class_labels) == {0, 1}
    assert result.feature_names == feature_names_4
    assert 0.0 <= result.oa  <= 1.0
    assert -1.0 <= result.kappa <= 1.0
    assert 0.0 <= result.minority_f1 <= 1.0
    assert isinstance(result.gate_passed, bool)
    assert isinstance(result.gate_message, str) and len(result.gate_message) > 0


def test_train_model_rf_cv_scores_length(balanced_data):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=5,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    assert len(result.cv_scores) == 3


def test_train_model_rf_feature_importances_sum_to_one(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)

    assert result.feature_importances is not None
    total = sum(result.feature_importances.values())
    assert pytest.approx(total, abs=1e-6) == 1.0


def test_train_model_rf_feature_importances_keys_match(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)
    assert set(result.feature_importances.keys()) == set(feature_names_4)


# ── train_model — XGBoost ────────────────────────────────────────────────────

def test_train_model_xgb_basic(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="xgboost", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)

    assert result.model_type == "xgboost"
    assert len(result.cv_scores) == 3
    assert result.feature_importances is not None
    assert pytest.approx(sum(result.feature_importances.values()), abs=1e-6) == 1.0


# ── train_model — Ensemble ────────────────────────────────────────────────────

def test_train_model_ensemble_model_type(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="ensemble", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)
    assert result.model_type == "ensemble"


def test_train_model_ensemble_importances_sum_to_one(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="ensemble", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)
    total  = sum(result.feature_importances.values())
    assert pytest.approx(total, abs=1e-6) == 1.0


# ── Per-class metrics ─────────────────────────────────────────────────────────

def test_per_class_metrics_has_required_keys(balanced_data):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    for cls in result.class_labels:
        assert set(result.per_class_metrics[cls].keys()) == {
            "precision", "recall", "f1", "support"
        }


def test_kappa_in_valid_range(balanced_data):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    assert -1.0 <= result.kappa <= 1.0


def test_minority_f1_is_min_over_non_majority():
    """_minority_f1 must return the minimum F1 across non-majority classes."""
    y = np.array([0] * 90 + [1] * 10, dtype=np.int32)   # majority = 0
    pcm = {
        0: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 90},
        1: {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 10},
    }
    assert _minority_f1(y, pcm) == pytest.approx(0.6)


# ── Gate integration via train_model ─────────────────────────────────────────

def test_train_model_gate_passes_on_separable_data(balanced_data, feature_names_4):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.80, "min_minority_f1": 0.70,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=50,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)
    assert result.gate_passed
    assert "passed" in result.gate_message.lower()


def test_train_model_gate_fails_with_tight_threshold(noise_data):
    """Random-label data should fail a 95% OA threshold."""
    X, y = noise_data
    cfg  = {"min_oa_threshold": 0.95, "min_minority_f1": 0.70,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    assert not result.gate_passed
    assert "failed" in result.gate_message.lower()


# ── SMOTE flag behaviour ──────────────────────────────────────────────────────

def test_smote_applied_when_forced_true(imbalanced_data, feature_names_4):
    X, y = imbalanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, apply_smote=True, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)
    assert result.smote_applied


def test_smote_not_applied_when_forced_false(imbalanced_data):
    X, y = imbalanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, apply_smote=False, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    assert not result.smote_applied


# ── ClassificationResult equality ────────────────────────────────────────────

def test_result_equality_excludes_model(balanced_data):
    """Two ClassificationResults must compare equal even with different models."""
    X, y  = balanced_data
    cfg   = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
             "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg  = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                  k_folds=3, random_state=0)
    r1    = train_model(X, y, ccfg, cfg=cfg)

    # Build a second result with a different model object but identical fields
    r2    = ClassificationResult(
        model=_build_model(ccfg),   # different (unfitted) model
        model_type=r1.model_type,
        class_labels=r1.class_labels,
        feature_names=r1.feature_names,
        oa=r1.oa,
        kappa=r1.kappa,
        per_class_metrics=r1.per_class_metrics,
        minority_f1=r1.minority_f1,
        feature_importances=r1.feature_importances,
        cv_scores=r1.cv_scores,
        smote_applied=r1.smote_applied,
        gate_passed=r1.gate_passed,
        gate_message=r1.gate_message,
    )
    assert r1 == r2


# ── Serialisation ─────────────────────────────────────────────────────────────

def test_model_is_picklable(balanced_data):
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    result = train_model(X, y, ccfg, cfg=cfg)
    dumped = pickle.dumps(result.model)
    loaded = pickle.loads(dumped)
    preds_orig   = result.model.predict(X[:5])
    preds_loaded = loaded.predict(X[:5])
    np.testing.assert_array_equal(preds_orig, preds_loaded)


# ── predict_raster ────────────────────────────────────────────────────────────

@pytest.fixture
def trained_rf(balanced_data, feature_names_4):
    """Small trained RF for raster prediction tests."""
    X, y = balanced_data
    cfg  = {"min_oa_threshold": 0.0, "min_minority_f1": 0.0,
            "default_k_folds": 3, "smote_auto_threshold_pct": 10}
    ccfg = ClassificationConfig(model_type="random_forest", n_estimators=10,
                                 k_folds=3, random_state=0)
    return train_model(X, y, ccfg, cfg=cfg, feature_names=feature_names_4)


def test_predict_raster_output_exists(trained_rf, make_raster, tmp_path):
    feat_path = make_raster("pred_feat.tif", bands=4, width=32, height=32,
                            dtype="float32")
    out       = tmp_path / "classified.tif"
    predict_raster(trained_rf.model, feat_path, out, trained_rf.feature_names)
    assert out.exists()


def test_predict_raster_output_shape_matches_input(trained_rf, make_raster, tmp_path):
    feat_path = make_raster("pred_feat2.tif", bands=4, width=48, height=48,
                            dtype="float32")
    out       = tmp_path / "classified2.tif"
    predict_raster(trained_rf.model, feat_path, out, trained_rf.feature_names)
    with rasterio.open(feat_path) as src, rasterio.open(out) as dst:
        assert dst.height == src.height
        assert dst.width  == src.width
        assert dst.count  == 1


def test_predict_raster_output_dtype_is_int16(trained_rf, make_raster, tmp_path):
    feat_path = make_raster("pred_feat3.tif", bands=4, width=32, height=32,
                            dtype="float32")
    out       = tmp_path / "classified3.tif"
    predict_raster(trained_rf.model, feat_path, out, trained_rf.feature_names)
    with rasterio.open(out) as ds:
        assert ds.dtypes[0] == "int16"


def test_predict_raster_nodata_pixels_remain_nodata(trained_rf, make_raster, tmp_path):
    """Pixels that are nodata in the feature stack must be nodata in the output."""
    feat_path = make_raster("pred_nd.tif", bands=4, width=32, height=32,
                            dtype="float32", nodata=-9999.0)
    # Set top-left 4×4 block to nodata in all feature bands
    with rasterio.open(feat_path, "r+") as ds:
        data = ds.read()
        data[:, :4, :4] = -9999.0
        ds.write(data)

    out      = tmp_path / "classified_nd.tif"
    out_nd   = -1
    predict_raster(trained_rf.model, feat_path, out, trained_rf.feature_names,
                   nodata=out_nd)

    with rasterio.open(out) as ds:
        classified = ds.read(1)
        nd         = ds.nodata
    # Every pixel in the nodata block must equal the output nodata value
    assert np.all(classified[:4, :4] == nd)
