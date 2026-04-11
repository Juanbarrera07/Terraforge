"""
Page 4 — ML Classification.

Uploads a label raster, configures training parameters, runs
extract_training_samples → train_model → predict_raster, displays the quality
gate, and unlocks Post-processing.

Session state written
---------------------
session["model"]      : ClassificationResult  — metrics + fitted model reference
session["classified"] : str                   — path to predicted raster on disk

Memory discipline
-----------------
X / y arrays from extract_training_samples are local to the training block and
released after train_model() returns.  The ClassificationResult stored in
session["model"] contains only the fitted model object (O(n_estimators) RAM) and
summary metrics — no pixel arrays.  The predicted raster lives on disk; its path
is stored in session state, not the pixel data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import audit, session
from pipeline.classify import (
    ClassificationConfig,
    ClassificationResult,
    extract_training_samples,
    predict_raster,
    train_model,
)
from ui._helpers import gate_metric, make_progress_cb, run_output_dir, save_upload


# ── Internal helpers ──────────────────────────────────────────────────────────

def _smote_option_to_bool(choice: str) -> bool | None:
    return {"Auto-detect": None, "Always on": True, "Always off": False}[choice]


def _render_gate(result: ClassificationResult, cfg: dict) -> None:
    """Display OA / Kappa / Minority F1 quality gate metrics."""
    min_oa = float(cfg.get("min_oa_threshold", 0.80))
    min_f1 = float(cfg.get("min_minority_f1",  0.70))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gate_metric("Overall Accuracy", result.oa, min_oa,
                    fmt=".4f", higher_is_better=True)
    with c2:
        gate_metric("Minority F1", result.minority_f1, min_f1,
                    fmt=".4f", higher_is_better=True)
    with c3:
        st.metric("Cohen's Kappa", f"{result.kappa:.4f}")
    with c4:
        smote_label = "Yes" if result.smote_applied else "No"
        st.metric("SMOTE applied", smote_label)

    if result.gate_passed:
        st.success(result.gate_message, icon="✅")
    else:
        st.error(result.gate_message, icon="❌")


def _render_importances(result: ClassificationResult) -> None:
    if not result.feature_importances:
        st.caption("Model does not expose feature importances.")
        return
    imp = pd.Series(result.feature_importances, name="Importance")
    imp = imp.sort_values(ascending=False)
    st.bar_chart(imp, use_container_width=True)
    with st.expander("Raw importance values"):
        st.dataframe(
            imp.reset_index().rename(columns={"index": "Feature"}).round(4),
            use_container_width=True, hide_index=True,
        )


def _render_per_class(result: ClassificationResult) -> None:
    rows = []
    for cls, m in result.per_class_metrics.items():
        rows.append({
            "Class":     cls,
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"],    4),
            "F1":        round(m["f1"],        4),
            "Support":   int(m["support"]),
        })
    st.dataframe(
        pd.DataFrame(rows).sort_values("Class"),
        use_container_width=True, hide_index=True,
    )


def _render_cv_scores(result: ClassificationResult) -> None:
    df = pd.DataFrame(
        {"Fold": range(1, len(result.cv_scores) + 1),
         "OA":   [round(s, 4) for s in result.cv_scores]},
    ).set_index("Fold")
    st.line_chart(df, use_container_width=True)
    st.caption(
        f"Mean fold OA: {sum(result.cv_scores)/len(result.cv_scores):.4f}  |  "
        f"Min: {min(result.cv_scores):.4f}  |  "
        f"Max: {max(result.cv_scores):.4f}"
    )


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("🤖 ML Classification")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return
    if not session.is_unlocked("classification"):
        st.warning(
            "🔒 **ML Classification is locked.**  "
            "Complete **Feature Engineering** first.",
            icon="⚠️",
        )
        return

    feat_result = session.get("features")
    if feat_result is None:
        st.warning(
            "No feature stack found. Return to **Feature Engineering**.", icon="⚠️"
        )
        return

    cfg         = session.get("config")
    out_dir     = run_output_dir(cfg, run_id, "classification")
    feature_path = feat_result.feature_path
    feat_names   = feat_result.feature_names

    st.caption(
        f"Feature stack: `{feature_path.name}`  |  "
        f"{len(feat_names)} features: {', '.join(feat_names)}"
    )

    # ── A: Label raster ───────────────────────────────────────────────────
    st.subheader("A — Label Raster")
    st.caption(
        "Upload a single-band integer raster with class labels. "
        "Must match the spatial extent of the feature stack."
    )

    st.info(
        "El label.tif debe tener exactamente el mismo CRS, resolución y extensión "
        "que el feature stack generado en el paso anterior. "
        "La app lo validará automáticamente al subirlo.",
        icon="📐",
    )

    with st.expander("ℹ️ ¿Qué debe ser el label.tif?"):
        st.markdown(
            """
| Proceso | Formato label.tif | Notas |
|---|---|---|
| **Clasificación supervisada (RF/XGBoost)** | 1 banda, dtype `int16`/`uint8`, valores enteros por clase | `0` = nodata, `1..N` = clases. Debe estar reproyectado al mismo CRS y resolución que el feature stack |
| **Detección de cambio** | No aplica — se usan dos runs consecutivos | El drift monitor compara áreas automáticamente |
| **Accuracy assessment** | CSV con columnas: `lat`, `lon`, `class_id` | Archivo separado del label.tif |
"""
        )

    label_mode = st.radio(
        "Input mode",
        options=["Upload file", "Local file path"],
        horizontal=True,
        key="label_input_mode",
        help=(
            "**Upload file** — browser upload (size-limited by Streamlit config).  "
            "**Local file path** — paste a path on the server; no size limit."
        ),
    )

    upload_dir = run_output_dir(cfg, run_id, "uploads")
    label_path: Path | None = None

    if label_mode == "Upload file":
        label_upload = st.file_uploader(
            "Label raster (.tif)", type=["tif", "tiff"], key="label_upload"
        )
        if label_upload is not None:
            label_path = save_upload(label_upload, upload_dir)
            st.success(f"Label raster saved: `{label_path.name}`", icon="✅")
        else:
            st.info("Upload a label raster to proceed.", icon="📂")
            # Re-use previously uploaded label from this session
            prev = session.get("_label_path")
            if prev and Path(prev).exists():
                label_path = Path(prev)
                st.caption(f"Re-using previous label: `{label_path.name}`")

    else:  # Local file path
        raw_path = st.text_input(
            "Label raster path",
            placeholder="/data/labels/training_labels.tif",
            key="label_local_path",
        )
        if raw_path and raw_path.strip():
            p = Path(raw_path.strip())
            if not p.exists():
                st.error(f"File not found — `{p}`", icon="❌")
            elif not p.is_file():
                st.error(f"Path is not a regular file — `{p}`", icon="❌")
            else:
                label_path = p
                st.success(f"Label raster found: `{label_path.name}`", icon="✅")
        else:
            st.info("Enter the server-side path to the label raster.", icon="📂")
            # Re-use previously resolved label path from this session
            prev = session.get("_label_path")
            if prev and Path(prev).exists():
                label_path = Path(prev)
                st.caption(f"Re-using previous label: `{label_path.name}`")

    if label_path is None:
        return

    # Persist label path within session (lightweight — just a string)
    session.set_("_label_path", str(label_path))

    # ── B: Training parameters ────────────────────────────────────────────
    st.divider()
    st.subheader("B — Training Parameters")

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Model type",
            options=["random_forest", "xgboost", "ensemble"],
            index=0,
            key="cls_model_type",
        )
        n_estimators = st.slider(
            "Number of estimators / trees",
            min_value=10, max_value=500, value=100, step=10,
            key="cls_n_estimators",
        )
        k_folds = st.slider(
            "CV folds", min_value=2, max_value=10, value=5,
            key="cls_k_folds",
        )
    with col2:
        smote_choice = st.radio(
            "SMOTE resampling",
            options=["Auto-detect", "Always on", "Always off"],
            index=0,
            key="cls_smote",
            help="Auto-detect triggers SMOTE when any class is < smote_threshold_pct% of total.",
        )
        max_depth = st.number_input(
            "Max tree depth (0 = unlimited)",
            min_value=0, max_value=50, value=0,
            key="cls_max_depth",
            help="0 means unlimited depth. Applies to RF and XGBoost.",
        )
        nodata_label = st.number_input(
            "Nodata label value",
            min_value=-999, max_value=999, value=0,
            key="cls_nodata_label",
            help="Integer pixel value in the label raster meaning 'not labeled'.",
        )

    st.markdown("**Sample cap**")
    col3, col4 = st.columns(2)
    with col3:
        use_max_samples = st.checkbox(
            "Limit training samples (max_samples)",
            value=False,
            key="cls_use_max_samples",
            help="Prevents memory issues on very large label rasters.",
        )
    with col4:
        max_samples_val = st.number_input(
            "Max samples",
            min_value=1000, max_value=10_000_000, value=100_000, step=10_000,
            key="cls_max_samples",
            disabled=not use_max_samples,
        )

    max_samples = int(max_samples_val) if use_max_samples else None

    class_cfg = ClassificationConfig(
        model_type          = model_type,
        n_estimators        = int(n_estimators),
        k_folds             = int(k_folds),
        apply_smote         = _smote_option_to_bool(smote_choice),
        random_state        = 42,
        max_depth           = int(max_depth) if max_depth > 0 else None,
    )

    # ── C: Train ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("C — Train & Predict")

    out_classified = out_dir / f"{run_id}_classified.tif"

    if st.button("▶ Extract Samples → Train → Predict", type="primary",
                 key="run_classification"):
        try:
            # Step 1 — extract samples (local arrays, freed after training)
            with st.spinner("Extracting training samples…"):
                X, y = extract_training_samples(
                    feature_path  = feature_path,
                    label_path    = label_path,
                    nodata_label  = int(nodata_label),
                    max_samples   = max_samples,
                    random_state  = 42,
                )
            st.info(
                f"Samples extracted: **{len(y):,}**  |  "
                f"Classes: {sorted(set(y.tolist()))}",
                icon="📊",
            )

            # Step 2 — train (X and y stay on the stack, not in session)
            with st.spinner(
                f"Training {model_type} with {k_folds}-fold CV…"
            ):
                result = train_model(
                    X, y,
                    class_cfg     = class_cfg,
                    cfg           = cfg,
                    feature_names = feat_names,
                )
            # Free training arrays immediately after model is built
            del X, y

            # Step 3 — predict raster (windowed, no full array in Python)
            with st.spinner("Predicting classification raster…"):
                predict_raster(
                    model         = result.model,
                    feature_path  = feature_path,
                    out_path      = out_classified,
                    feature_names = feat_names,
                )

            # Persist result and classified path (path only, not pixel data)
            session.set_("model",      result)
            session.set_("classified", str(out_classified))

            audit.log_event(
                run_id, "gate",
                {
                    "stage":       "classification",
                    "model_type":  result.model_type,
                    "oa":          round(result.oa, 4),
                    "kappa":       round(result.kappa, 4),
                    "minority_f1": round(result.minority_f1, 4),
                    "smote":       result.smote_applied,
                    "output":      str(out_classified),
                },
                decision="proceed" if result.gate_passed else "block",
            )
            st.success("Training and prediction complete.", icon="✅")

        except ValueError as exc:
            st.error(str(exc), icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "classification", "error": str(exc)})
        except Exception as exc:
            st.error(f"Classification failed: {exc}", icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "classification", "error": str(exc)})

    # ── D: Results ────────────────────────────────────────────────────────
    result: ClassificationResult | None = session.get("model")
    if result is None:
        return

    st.divider()
    st.subheader("D — Quality Gate")
    _render_gate(result, cfg)

    st.divider()
    st.subheader("E — Feature Importances")
    _render_importances(result)

    st.divider()
    st.subheader("F — Per-class Metrics")
    _render_per_class(result)

    st.divider()
    st.subheader("G — Cross-validation OA per Fold")
    _render_cv_scores(result)

    # ── E: Unlock gate ────────────────────────────────────────────────────
    st.divider()
    if session.is_unlocked("postprocess"):
        st.success("Post-processing already unlocked.", icon="✅")
        return

    classified_path = session.get("classified")
    if not classified_path:
        return

    if result.gate_passed:
        st.success("Quality gate passed. Proceed to post-processing.", icon="✅")
        if st.button("🔓 Unlock Post-processing →", type="primary"):
            session.unlock_stage("postprocess")
            audit.log_event(
                run_id, "decision",
                {"action": "unlock_postprocess",
                 "classified_path": classified_path},
                decision="proceed",
            )
            st.rerun()
    else:
        st.error(
            "**Quality gate failed.**  Review metrics above or override below.",
            icon="🚫",
        )
        override = st.checkbox(
            "⚠️ Override quality gate and proceed anyway",
            key="cls_gate_override",
        )
        if override:
            st.warning(
                "Gate override active — results may not meet accuracy standards.",
                icon="⚠️",
            )
            if st.button("🔓 Unlock Post-processing (override) →"):
                session.unlock_stage("postprocess")
                audit.log_event(
                    run_id, "decision",
                    {
                        "action":          "override_classification_gate",
                        "oa":              round(result.oa, 4),
                        "minority_f1":     round(result.minority_f1, 4),
                        "gate_message":    result.gate_message,
                        "classified_path": classified_path,
                    },
                    decision="override",
                )
                st.rerun()
