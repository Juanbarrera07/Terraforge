"""
Phase 5 — ML classification.

Provides training sample extraction, class imbalance detection, SMOTE
resampling, k-fold cross-validation, and raster prediction for three
classifier backends: Random Forest, XGBoost, and an explicit Ensemble.

Public API
----------
ClassificationConfig  — algorithm parameters (not quality thresholds)
ClassificationResult  — frozen dataclass returned by train_model()
extract_training_samples — windowed extraction from feature + label rasters
detect_imbalance         — per-class percentage check
apply_smote              — SMOTE resampling wrapper
train_model              — CV training + quality gate
predict_raster           — windowed tile-by-tile raster prediction
apply_quality_gate       — standalone gate (also used internally by train_model)

Design rules
------------
- No Streamlit imports.
- All raster reads through iter_windows; no list(iter_windows(...)).
- Memory in extract_training_samples scales with labeled pixel count, not
  raster size.  max_samples imposes a hard cap via stratified subsampling.
- SMOTE is applied inside each CV fold (not before splitting) to prevent
  data leakage.
- Feature importances are always normalised to sum to 1.0 across models.
- Model field is excluded from ClassificationResult equality.
- Quality gate thresholds come from the cfg dict; no hardcoded values.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from pipeline.config_loader import load_config
from pipeline.raster_io import DEFAULT_BLOCK, iter_windows


# ── Config dataclass ──────────────────────────────────────────────────────────

@dataclass
class ClassificationConfig:
    """
    Algorithm-level parameters for a classification run.

    Quality-gate thresholds (min_oa_threshold, min_minority_f1) are read from
    the pipeline cfg dict, not stored here.

    Fields
    ------
    model_type           : "random_forest" | "xgboost" | "ensemble"
    n_estimators         : Number of trees / boosting rounds.
    k_folds              : CV folds; overridden by cfg["default_k_folds"].
    apply_smote          : True / False / None.  None = auto-detect via
                           smote_threshold_pct.
    smote_threshold_pct  : Class percentage below which auto-SMOTE triggers;
                           overridden by cfg["smote_auto_threshold_pct"].
    random_state         : RNG seed (affects CV splits, SMOTE, model init).
    max_depth            : Tree max depth; None = unlimited (RF) / 6 (XGB).
    """
    model_type:          str           = "random_forest"
    n_estimators:        int           = 100
    k_folds:             int           = 5
    apply_smote:         Optional[bool] = None
    smote_threshold_pct: float         = 10.0
    random_state:        int           = 42
    max_depth:           Optional[int] = None


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassificationResult:
    """
    Immutable result from train_model().

    ``model`` is excluded from equality comparisons (consistent with
    CoregistrationResult.shift_map and FeatureResult.correlation_matrix).

    Fields
    ------
    model                : Trained sklearn / XGBoost estimator.
    model_type           : "random_forest" | "xgboost" | "ensemble"
    class_labels         : Sorted list of class integers seen in training data.
    feature_names        : Feature names matching training column order.
    oa                   : Overall accuracy from OOF CV predictions.
    kappa                : Cohen's kappa from OOF CV predictions.
    per_class_metrics    : {class_label: {precision, recall, f1, support}}
    minority_f1          : Min F1 across non-majority classes.
    feature_importances  : {feature_name: importance} normalised to sum=1.0,
                           or None if the model does not expose importances.
    cv_scores            : Per-fold OA (len == k_folds).
    smote_applied        : Whether SMOTE was actually used.
    gate_passed          : True if OA and minority_f1 both meet thresholds.
    gate_message         : Human-readable gate outcome.
    """
    model:               Any                              = field(compare=False)
    model_type:          str
    class_labels:        list[int]
    feature_names:       list[str]
    oa:                  float
    kappa:               float
    per_class_metrics:   dict[int, dict[str, float]]
    minority_f1:         float
    feature_importances: Optional[dict[str, float]]
    cv_scores:           list[float]
    smote_applied:       bool
    gate_passed:         bool
    gate_message:        str


# ── Private: helpers ──────────────────────────────────────────────────────────

def _stratified_subsample(
    X:            np.ndarray,
    y:            np.ndarray,
    max_samples:  int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified random subsample — proportional allocation per class.

    Each class receives ``round(max_samples * class_pct)`` samples, with a
    minimum of 1 per class.  Total may differ from max_samples by at most
    ``n_classes`` due to rounding.
    """
    rng     = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    n_total = len(y)
    keep: list[np.ndarray] = []

    for cls, cnt in zip(classes, counts):
        idx    = np.where(y == cls)[0]
        budget = max(1, round(max_samples * int(cnt) / n_total))
        budget = min(budget, int(cnt))
        keep.append(rng.choice(idx, size=budget, replace=False))

    idx_all = np.concatenate(keep)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


def _build_model(class_cfg: ClassificationConfig) -> Any:
    """Return an unfitted estimator for the requested model_type."""
    rf = RandomForestClassifier(
        n_estimators=class_cfg.n_estimators,
        max_depth=class_cfg.max_depth,
        random_state=class_cfg.random_state,
        n_jobs=-1,
    )
    if class_cfg.model_type == "random_forest":
        return rf

    xgb = XGBClassifier(
        n_estimators=class_cfg.n_estimators,
        max_depth=class_cfg.max_depth or 6,
        random_state=class_cfg.random_state,
        verbosity=0,
    )
    if class_cfg.model_type == "xgboost":
        return xgb

    if class_cfg.model_type == "ensemble":
        return VotingClassifier(
            estimators=[("rf", rf), ("xgb", xgb)],
            voting="soft",
        )

    raise ValueError(
        f"Unknown model_type {class_cfg.model_type!r}. "
        "Expected 'random_forest', 'xgboost', or 'ensemble'."
    )


def _compute_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    class_labels: list[int],
) -> dict:
    """Compute OA, kappa, and per-class precision/recall/F1/support."""
    oa    = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=class_labels,
        zero_division=0,
    )

    per_class: dict[int, dict[str, float]] = {}
    for i, cls in enumerate(class_labels):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }

    return {"oa": oa, "kappa": kappa, "per_class_metrics": per_class}


def _minority_f1(
    y:                  np.ndarray,
    per_class_metrics:  dict[int, dict[str, float]],
) -> float:
    """
    Return the minimum F1 score across non-majority classes.

    The majority class is determined by sample count in ``y``.
    Returns 1.0 if there is only one class (degenerate / binary same-class case).
    """
    counts   = Counter(y.tolist())
    majority = max(counts, key=counts.get)  # type: ignore[arg-type]

    non_majority = [
        metrics["f1"]
        for cls, metrics in per_class_metrics.items()
        if cls != majority
    ]
    return min(non_majority) if non_majority else 1.0


def _extract_feature_importances(
    model:         Any,
    model_type:    str,
    feature_names: list[str],
) -> Optional[dict[str, float]]:
    """
    Extract feature importances and normalise to sum = 1.0.

    For "ensemble", average the normalised importances from each constituent.
    Returns None if the model does not expose ``feature_importances_``.
    """
    try:
        if model_type == "ensemble":
            parts: list[np.ndarray] = []
            for est in model.estimators_:
                raw   = np.asarray(est.feature_importances_, dtype=np.float64)
                total = raw.sum()
                parts.append(raw / total if total > 1e-12 else raw)
            avg = np.mean(parts, axis=0)
        else:
            raw   = np.asarray(model.feature_importances_, dtype=np.float64)
            total = raw.sum()
            avg   = raw / total if total > 1e-12 else raw

        return {name: float(v) for name, v in zip(feature_names, avg)}
    except AttributeError:
        return None


# ── Public functions ──────────────────────────────────────────────────────────

def extract_training_samples(
    feature_path:  str | Path,
    label_path:    str | Path,
    nodata_label:  int           = 0,
    max_samples:   Optional[int] = None,
    random_state:  int           = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract labeled training samples from a feature stack and a label raster.

    Memory model
    ------------
    Pixels are collected tile-by-tile via windowed reads; only labeled pixels
    are retained, so memory scales with the labeled pixel count, not the full
    raster size.  When max_samples is set, a stratified subsample is applied
    after collection, providing a hard cap on the returned array size.

    Parameters
    ----------
    feature_path : Feature stack GeoTIFF (output of features.compute_features).
    label_path   : Single-band integer raster.  Each pixel value is a class
                   label; ``nodata_label`` marks unlabeled pixels.
                   The raster's file-level nodata is also treated as unlabeled.
    nodata_label : Integer pixel value meaning "not labeled".  Default 0.
                   Must not equal any valid class label.
    max_samples  : If set, stratified-randomly subsample to at most this many
                   samples after collection.  Proportional per class.
    random_state : RNG seed used for optional subsampling.

    Returns
    -------
    X : (n_samples, n_features) float32 — feature vectors.
    y : (n_samples,)            int32   — class labels.

    Raises
    ------
    ValueError : Raster shape mismatch, or no labeled samples found.
    """
    feature_path = Path(feature_path)
    label_path   = Path(label_path)

    with rasterio.open(feature_path) as fsrc, rasterio.open(label_path) as lsrc:
        if (fsrc.height, fsrc.width) != (lsrc.height, lsrc.width):
            raise ValueError(
                f"Raster shape mismatch: feature ({fsrc.height}×{fsrc.width}) "
                f"vs label ({lsrc.height}×{lsrc.width})."
            )
        n_features  = fsrc.count
        feat_nodata = fsrc.nodata
        lab_nodata  = lsrc.nodata

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for window in iter_windows(fsrc, DEFAULT_BLOCK):
            feat = fsrc.read(window=window).astype(np.float32)   # (n_feat, H, W)
            lab  = lsrc.read(1, window=window)                   # (H, W)

            # Valid label: not the nodata_label sentinel and not file-level nodata
            valid_lab = lab != nodata_label
            if lab_nodata is not None:
                valid_lab &= (lab != lab_nodata)

            # Valid features: all bands finite and not feat_nodata
            valid_feat = np.ones(lab.shape, dtype=bool)
            for b in range(n_features):
                valid_feat &= np.isfinite(feat[b])
                if feat_nodata is not None:
                    valid_feat &= feat[b] != feat_nodata

            mask = valid_lab & valid_feat
            if not mask.any():
                continue

            # feat[:, mask] → (n_feat, n_valid); .T → (n_valid, n_feat)
            X_parts.append(feat[:, mask].T)
            y_parts.append(lab[mask].astype(np.int32))

    if not X_parts:
        raise ValueError(
            f"No labeled training samples found. "
            f"Verify the label raster has pixels with values other than "
            f"nodata_label={nodata_label}."
        )

    X = np.concatenate(X_parts, axis=0)   # (n_total, n_features) — C-contiguous
    y = np.concatenate(y_parts, axis=0)   # (n_total,)

    if max_samples is not None and len(y) > max_samples:
        X, y = _stratified_subsample(X, y, max_samples, random_state)

    return X, y


def detect_imbalance(
    y:             np.ndarray,
    threshold_pct: float,
) -> tuple[bool, dict[int, float]]:
    """
    Report per-class sample percentages and flag imbalanced datasets.

    Parameters
    ----------
    y             : 1-D integer label array.
    threshold_pct : A class is considered under-represented when its
                    percentage of total samples is strictly below this value.
                    pct_of_total = class_count / total_samples * 100.

    Returns
    -------
    (is_imbalanced, class_pct)
        is_imbalanced : True if any class percentage < threshold_pct.
        class_pct     : {class_label: pct_of_total} — values sum to 100.
    """
    counts    = Counter(y.tolist())
    n_total   = len(y)
    class_pct = {cls: cnt / n_total * 100.0 for cls, cnt in counts.items()}
    is_imb    = any(pct < threshold_pct for pct in class_pct.values())
    return is_imb, class_pct


def apply_smote(
    X:            np.ndarray,
    y:            np.ndarray,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling to balance the class distribution.

    SMOTE requires at least k_neighbors + 1 (default: 6) samples per
    minority class.  Raises ValueError from imblearn if this is not met.

    Returns
    -------
    X_resampled, y_resampled — both C-contiguous numpy arrays.
    """
    from imblearn.over_sampling import SMOTE  # lazy import — optional dep

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return np.ascontiguousarray(X_res), np.ascontiguousarray(y_res)


def apply_quality_gate(
    oa:          float,
    minority_f1: float,
    cfg:         dict,
) -> tuple[bool, str]:
    """
    Evaluate the classification quality gate.

    Gate passes iff:
        oa          >= cfg["min_oa_threshold"]
        minority_f1 >= cfg["min_minority_f1"]

    Parameters
    ----------
    oa, minority_f1 : Metrics from OOF CV evaluation.
    cfg             : Pipeline config dict (must contain the two threshold keys).

    Returns
    -------
    (gate_passed: bool, gate_message: str)
    """
    min_oa = float(cfg.get("min_oa_threshold", 0.80))
    min_f1 = float(cfg.get("min_minority_f1",  0.70))

    failures: list[str] = []
    if oa < min_oa:
        failures.append(f"OA {oa:.3f} < threshold {min_oa:.3f}")
    if minority_f1 < min_f1:
        failures.append(f"minority F1 {minority_f1:.3f} < threshold {min_f1:.3f}")

    if failures:
        return False, "Gate failed: " + "; ".join(failures)
    return True, (
        f"Gate passed: OA {oa:.3f} >= {min_oa:.3f}, "
        f"minority F1 {minority_f1:.3f} >= {min_f1:.3f}"
    )


def train_model(
    X:             np.ndarray,
    y:             np.ndarray,
    class_cfg:     ClassificationConfig,
    cfg:           Optional[dict]       = None,
    feature_names: Optional[list[str]] = None,
) -> ClassificationResult:
    """
    Train a classifier with stratified k-fold CV and apply the quality gate.

    Algorithm
    ---------
    1. Resolve config overrides (k_folds, smote_threshold_pct from cfg).
    2. Determine SMOTE: explicit bool, or auto-detect via detect_imbalance.
    3. Run StratifiedKFold CV:
       - SMOTE applied inside each fold to the training split only.
       - SMOTE failures (too few minority samples) are caught and logged.
       - OOF predictions collected across all folds.
    4. Compute OA, kappa, per-class metrics from full OOF predictions.
    5. Fit final model on full (X, y) — with SMOTE if applicable.
    6. Extract and normalise feature importances.
    7. Apply quality gate; return ClassificationResult.

    Parameters
    ----------
    X             : (n_samples, n_features) float32 feature array.
    y             : (n_samples,) int32 label array.
    class_cfg     : Algorithm parameters.
    cfg           : Pipeline config dict; loaded from pipeline_config.yaml if None.
    feature_names : Names for the n_features columns.  Defaults to ["0","1",...].
                    Used in feature_importances keys and ClassificationResult.

    Returns
    -------
    ClassificationResult
    """
    # Callers with an existing config pass it directly; load_config() is
    # the fallback only.
    if cfg is None:
        cfg = load_config()

    # Config overrides for CV and SMOTE parameters
    k_folds     = int(cfg.get("default_k_folds",          class_cfg.k_folds))
    smote_thr   = float(cfg.get("smote_auto_threshold_pct", class_cfg.smote_threshold_pct))

    class_labels  = sorted(int(c) for c in np.unique(y))
    n_features    = X.shape[1]
    feat_names    = feature_names if feature_names is not None else [str(i) for i in range(n_features)]

    # Resolve SMOTE decision
    if class_cfg.apply_smote is None:
        is_imb, _ = detect_imbalance(y, smote_thr)
        do_smote  = is_imb
    else:
        do_smote = class_cfg.apply_smote

    base_model = _build_model(class_cfg)

    # ── Stratified k-fold CV with OOF collection ──────────────────────────────
    skf         = StratifiedKFold(n_splits=k_folds, shuffle=True,
                                  random_state=class_cfg.random_state)
    oof_true    = np.empty(len(y), dtype=np.int32)
    oof_pred    = np.empty(len(y), dtype=np.int32)
    cv_scores: list[float] = []
    smote_applied = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if do_smote:
            try:
                X_tr, y_tr = apply_smote(X_tr, y_tr, class_cfg.random_state)
                smote_applied = True
            except Exception as exc:
                warnings.warn(
                    f"SMOTE failed in fold {fold} ({exc}); "
                    "proceeding without resampling for this fold.",
                    stacklevel=2,
                )

        fold_model = clone(base_model)
        fold_model.fit(X_tr, y_tr)
        y_hat = fold_model.predict(X_val).astype(np.int32)

        oof_true[val_idx] = y_val
        oof_pred[val_idx] = y_hat
        cv_scores.append(float(accuracy_score(y_val, y_hat)))

    # ── Metrics from aggregated OOF predictions ───────────────────────────────
    metrics   = _compute_metrics(oof_true, oof_pred, class_labels)
    oa        = metrics["oa"]
    kappa     = metrics["kappa"]
    pcm       = metrics["per_class_metrics"]
    min_f1    = _minority_f1(y, pcm)

    # ── Final model fit on full data ──────────────────────────────────────────
    X_final, y_final = X, y
    if do_smote and smote_applied:
        try:
            X_final, y_final = apply_smote(X, y, class_cfg.random_state)
        except Exception as exc:
            warnings.warn(
                f"SMOTE failed on final fit ({exc}); training on original data.",
                stacklevel=2,
            )
    final_model = clone(base_model)
    final_model.fit(X_final, y_final)

    # ── Feature importances (normalised) ─────────────────────────────────────
    feat_imp = _extract_feature_importances(
        final_model, class_cfg.model_type, feat_names
    )

    # ── Quality gate ──────────────────────────────────────────────────────────
    gate_passed, gate_message = apply_quality_gate(oa, min_f1, cfg)

    return ClassificationResult(
        model=final_model,
        model_type=class_cfg.model_type,
        class_labels=class_labels,
        feature_names=feat_names,
        oa=oa,
        kappa=kappa,
        per_class_metrics=pcm,
        minority_f1=min_f1,
        feature_importances=feat_imp,
        cv_scores=cv_scores,
        smote_applied=smote_applied,
        gate_passed=gate_passed,
        gate_message=gate_message,
    )


def predict_raster(
    model:         Any,
    feature_path:  str | Path,
    out_path:      str | Path,
    feature_names: list[str],
    block_size:    int = DEFAULT_BLOCK,
    nodata:        int = -1,
) -> Path:
    """
    Apply a trained classifier to a feature stack raster, tile by tile.

    Each tile is predicted independently; no full-raster arrays are created.
    Pixels where any feature band is nodata or non-finite are written as
    ``nodata`` in the output without calling the model.

    Parameters
    ----------
    model         : Trained sklearn / XGBoost estimator.
    feature_path  : Feature stack GeoTIFF (from features.compute_features).
    out_path      : Output classified raster path.
    feature_names : Feature names in band order — must match the order used
                    during training (used for metadata only; not reordered here).
    block_size    : Tile side length for windowed processing.
    nodata        : Output nodata sentinel.  Must not equal any valid class label.
                    Written as int16; default -1.

    Returns
    -------
    Path of the written classified raster.
    """
    feature_path = Path(feature_path)
    out_path     = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_path) as src:
        feat_nodata = src.nodata
        out_profile = src.profile.copy()
        n_features  = src.count

    out_profile.update({
        "count":      1,
        "dtype":      "int16",
        "nodata":     nodata,
        "compress":   "deflate",
        "predictor":  2,    # horizontal integer differencing
        "tiled":      True,
        "blockxsize": block_size,
        "blockysize": block_size,
    })

    with (
        rasterio.open(feature_path) as src,
        rasterio.open(out_path, "w", **out_profile) as dst,
    ):
        for window in iter_windows(src, block_size):
            tile = src.read(window=window).astype(np.float32)   # (n_feat, H, W)
            H, W = tile.shape[1], tile.shape[2]

            # Per-pixel validity: all feature bands must be finite and non-nodata
            valid = np.ones((H, W), dtype=bool)
            for b in range(n_features):
                valid &= np.isfinite(tile[b])
                if feat_nodata is not None:
                    valid &= tile[b] != feat_nodata

            out_tile = np.full((H, W), nodata, dtype=np.int16)

            if valid.any():
                # tile[:, valid] → (n_feat, n_valid)
                # .T             → (n_valid, n_feat) — Fortran order view
                # .copy()        → C-contiguous — one explicit copy, no hidden copies
                X_valid  = tile[:, valid].T.copy()
                preds    = model.predict(X_valid).astype(np.int16)
                out_tile[valid] = preds

            dst.write(out_tile[np.newaxis], window=window)

    return out_path
