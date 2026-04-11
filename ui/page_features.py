"""
Page 3 — Feature Engineering.

Builds a per-pixel feature stack (spectral indices, SAR ratio, terrain
derivatives) from a preprocessed raster.  The user configures the band map
and controls which individual features to include.

Session state written
---------------------
session["features"]  : FeatureResult  — contains feature_path (on disk),
                       feature_names, feature_stats, texture_summary,
                       correlation_matrix, high_correlation_pairs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import audit, session
from pipeline.features import BandMap, FeatureResult, active_features, compute_features
from pipeline.raster_io import get_meta
from ui._helpers import gate_metric, make_progress_cb, run_output_dir


# ── Feature metadata ──────────────────────────────────────────────────────────

_FEATURE_LABELS: dict[str, str] = {
    "ndvi":      "NDVI — Normalised Difference Vegetation Index (NIR, Red)",
    "ndwi":      "NDWI — Normalised Difference Water Index (Green, NIR)",
    "bsi":       "BSI  — Bare Soil Index (SWIR, Red, NIR, Blue)",
    "ndre":      "NDRE — Normalised Difference Red-Edge (RedEdge, Red)",
    "vari":      "VARI — Visible Atmospherically Resistant Index (Green, Red, Blue)",
    "sar_ratio": "SAR Ratio — VV / VH backscatter ratio",
    "slope":     "Slope — terrain gradient from DEM",
    "aspect":    "Aspect — terrain aspect from DEM",
}

_BAND_FIELDS: list[tuple[str, str]] = [
    ("nir",     "NIR band"),
    ("red",     "Red band"),
    ("green",   "Green band"),
    ("blue",    "Blue band"),
    ("swir",    "SWIR band"),
    ("rededge", "Red-Edge band"),
    ("vv",      "SAR VV band"),
    ("vh",      "SAR VH band"),
    ("dem",     "DEM band"),
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _candidate_sources(raw_data: dict, preprocessed: dict) -> dict[str, Path]:
    """
    Return {display_label: path} for all available source rasters.

    Preprocessed outputs take priority over raw inputs for the same layer key.
    """
    sources: dict[str, Path] = {}
    for key, layer in raw_data.items():
        name  = key.split("__")[-1]
        p_raw = Path(str(layer["path"]))
        if preprocessed and key in preprocessed:
            p     = Path(preprocessed[key])
            label = f"[preprocessed]  {name}"
        else:
            p     = p_raw
            label = f"[raw]  {name}"
        sources[label] = p
    return sources


def _band_map_ui(n_bands: int) -> BandMap:
    """
    Render band-mapping selectboxes for all BandMap fields.
    Returns a BandMap with 1-indexed band numbers or None per field.
    """
    st.markdown("**Band mapping** — assign raster bands to logical roles.")
    st.caption(
        f"Source raster has **{n_bands}** band{'s' if n_bands != 1 else ''}.  "
        "Set a field to *None* to skip features that require it."
    )

    band_options = ["None"] + [str(i) for i in range(1, n_bands + 1)]

    col1, col2 = st.columns(2)
    values: dict[str, int | None] = {}

    for idx, (field_name, label) in enumerate(_BAND_FIELDS):
        col = col1 if idx % 2 == 0 else col2
        with col:
            sel = st.selectbox(
                label,
                options=band_options,
                index=0,
                key=f"band_{field_name}",
            )
        values[field_name] = None if sel == "None" else int(sel)

    return BandMap(**values)


def _feature_toggle_ui(available: list[str]) -> list[str]:
    """
    Render per-feature enable/disable checkboxes.
    Returns the list of enabled feature names.
    """
    if not available:
        return []

    st.markdown("**Feature selection** — enable or disable individual features.")
    st.caption(
        "Only features whose required bands are mapped above are shown. "
        "Disabling a feature reduces the output stack size."
    )

    enabled: list[str] = []
    for name in available:
        label = _FEATURE_LABELS.get(name, name)
        if st.checkbox(label, value=True, key=f"feat_en_{name}"):
            enabled.append(name)

    return enabled


def _render_feature_stats(result: FeatureResult) -> None:
    """Display per-feature statistics as a dataframe."""
    rows = []
    for name, stats in result.feature_stats.items():
        rows.append({
            "Feature":    name,
            "Min":        f"{stats['min']:.4f}",
            "Max":        f"{stats['max']:.4f}",
            "Mean":       f"{stats['mean']:.4f}",
            "Std":        f"{stats['std']:.4f}",
            "Valid %":    f"{stats['valid_pct']:.1f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_correlation(result: FeatureResult) -> None:
    """Warn on high-correlation pairs; show full matrix in expander."""
    cfg         = session.get("config") or {}
    corr_thresh = float(cfg.get("corr_flag_threshold", 0.95))

    if result.high_correlation_pairs:
        st.warning(
            f"**{len(result.high_correlation_pairs)} high-correlation pair(s) "
            f"detected** (|r| > {corr_thresh}).  "
            "Consider dropping redundant features before training.",
            icon="⚠️",
        )
        for a, b, r in result.high_correlation_pairs:
            st.markdown(f"  • **{a}** ↔ **{b}**  —  r = {r:.4f}")
    else:
        st.success(
            f"No feature pairs with |r| > {corr_thresh}.",
            icon="✅",
        )

    with st.expander("Full correlation matrix"):
        names = result.feature_names
        df    = pd.DataFrame(
            result.correlation_matrix,
            index=names,
            columns=names,
        ).round(3)
        st.dataframe(df, use_container_width=True)


def _render_texture_summary(result: FeatureResult) -> None:
    if result.texture_summary is None:
        return
    with st.expander("GLCM texture summary (tile-level aggregates)"):
        rows = []
        for stat_name, vals in result.texture_summary.items():
            rows.append({
                "Statistic": stat_name,
                "Mean":  f"{vals['mean']:.4f}",
                "Std":   f"{vals['std']:.4f}",
                "Min":   f"{vals['min']:.4f}",
                "Max":   f"{vals['max']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("🔬 Feature Engineering")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return
    if not session.is_unlocked("features"):
        st.warning(
            "🔒 **Feature Engineering is locked.**  "
            "Complete **Preprocessing & Coregistration** first.",
            icon="⚠️",
        )
        return

    raw_data:     dict | None = session.get("raw_data")
    preprocessed: dict        = session.get("preprocessed") or {}

    if not raw_data:
        st.warning("No layers available. Return to **Data Ingestion**.", icon="⚠️")
        return

    cfg     = session.get("config")
    out_dir = run_output_dir(cfg, run_id, "features")

    # ── Source raster selection ────────────────────────────────────────────
    st.subheader("A — Source Raster")
    sources = _candidate_sources(raw_data, preprocessed)
    if not sources:
        st.warning("No raster files found.", icon="⚠️")
        return

    src_label = st.selectbox(
        "Select source raster for feature computation",
        options=list(sources.keys()),
        key="feat_src_select",
    )
    src_path = sources[src_label]

    # Read metadata to drive band count
    try:
        meta    = get_meta(src_path)
        n_bands = meta["count"]
        st.caption(
            f"Path: `{src_path.name}`  |  "
            f"{meta['width']}×{meta['height']} px  |  "
            f"{n_bands} band{'s' if n_bands != 1 else ''}  |  "
            f"CRS: {meta.get('crs_epsg', 'unknown')}"
        )
    except Exception as exc:
        st.error(f"Cannot read raster metadata: {exc}", icon="❌")
        return

    # ── Band map UI ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("B — Band Mapping")
    band_map = _band_map_ui(n_bands)

    # Preview available features from the current band map
    available = active_features(band_map)

    if not available:
        st.warning(
            "The current band mapping does not support any features.  "
            "Assign at least two band fields (e.g. NIR + Red for NDVI).",
            icon="⚠️",
        )
        # Still allow the user to edit band map; stop here
        return

    # ── Feature selection ──────────────────────────────────────────────────
    st.divider()
    st.subheader("C — Feature Selection")
    enabled = _feature_toggle_ui(available)

    if not enabled:
        st.warning("No features enabled. Enable at least one feature.", icon="⚠️")
        return

    st.caption(f"{len(enabled)} feature(s) selected: {', '.join(enabled)}")

    # ── Run ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("D — Compute Features")

    out_path = out_dir / f"{run_id}_feature_stack.tif"

    if st.button("▶ Compute Feature Stack", type="primary", key="run_features"):
        bar = st.progress(0, text="Computing features…")
        try:
            with st.spinner("Pass 1 — computing and writing feature stack…"):
                result = compute_features(
                    src_path         = src_path,
                    out_path         = out_path,
                    band_map         = band_map,
                    progress         = make_progress_cb(bar),
                    cfg              = cfg,
                    enabled_features = enabled,
                )
            bar.progress(1.0, text="Done.")
            session.set_("features", result)

            audit.log_event(
                run_id, "gate",
                {
                    "stage":            "feature_engineering",
                    "features":         result.feature_names,
                    "n_high_corr_pairs": len(result.high_correlation_pairs),
                    "output":           str(result.feature_path),
                },
                decision="proceed",
            )
            st.success(
                f"Feature stack written to `{out_path.name}` "
                f"({len(result.feature_names)} features).",
                icon="✅",
            )
        except ValueError as exc:
            bar.empty()
            st.error(str(exc), icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "feature_engineering", "error": str(exc)})
        except Exception as exc:
            bar.empty()
            st.error(f"Feature computation failed: {exc}", icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "feature_engineering", "error": str(exc)})

    # ── Results ────────────────────────────────────────────────────────────
    feat_result: FeatureResult | None = session.get("features")
    if feat_result is None:
        return

    st.divider()
    st.subheader("E — Results")

    # Summary metrics
    n_feats  = len(feat_result.feature_names)
    n_high_r = len(feat_result.high_correlation_pairs)
    col1, col2, col3 = st.columns(3)
    col1.metric("Features computed", n_feats)
    col2.metric("High-corr pairs",   n_high_r,
                delta=None if n_high_r == 0 else f"{n_high_r} flagged",
                delta_color="off" if n_high_r == 0 else "inverse")
    col3.metric("Output path", feat_result.feature_path.name)

    # Feature stats table
    st.markdown("**Per-feature statistics**")
    _render_feature_stats(feat_result)

    # Correlation
    st.markdown("**Feature correlation**")
    _render_correlation(feat_result)

    # GLCM
    _render_texture_summary(feat_result)

    # ── Unlock gate ────────────────────────────────────────────────────────
    st.divider()
    if session.is_unlocked("classification"):
        st.success("ML Classification already unlocked.", icon="✅")
        return

    st.success("Feature stack ready. Proceed to classification.", icon="✅")
    if st.button("🔓 Unlock ML Classification →", type="primary"):
        session.unlock_stage("classification")
        audit.log_event(
            run_id, "decision",
            {"action": "unlock_classification",
             "feature_path": str(feat_result.feature_path),
             "n_features": len(feat_result.feature_names)},
            decision="proceed",
        )
        st.success("ML Classification unlocked!")
        st.rerun()
