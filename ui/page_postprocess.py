"""
Page 5 — Post-processing & Validation.

Applies spatial filters (sieve MMU, morphological close), computes class areas
with drift detection, and runs accuracy assessment from a reference CSV.

Session state written
---------------------
session["classified"]          : str            — updated to postprocessed path
session["class_areas"]         : ClassAreaResult — current run areas
session["previous_class_areas"]: dict           — saved reference areas (ha)
session["accuracy"]            : AccuracyResult  — from reference CSV
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import audit, session
from pipeline.postprocess import (
    AccuracyResult,
    ClassAreaResult,
    DriftResult,
    assess_accuracy_from_points,
    check_drift,
    compute_class_areas,
    morphological_close,
    sieve_filter,
)
from pipeline.raster_io import get_meta
from ui._helpers import gate_metric, run_output_dir, save_upload


# ── Internal helpers ──────────────────────────────────────────────────────────

def _areas_dataframe(areas: ClassAreaResult) -> pd.DataFrame:
    """Build a display DataFrame with ha and % columns."""
    rows = []
    for cls_id in areas.class_ids:
        ha  = areas.areas_ha[cls_id]
        pct = (ha / areas.total_area_ha * 100.0) if areas.total_area_ha > 0 else 0.0
        rows.append({
            "Class":    cls_id,
            "Pixels":   areas.pixel_counts[cls_id],
            "Area (ha)": round(ha,  4),
            "Area (%)": round(pct, 2),
        })
    return pd.DataFrame(rows)


def _render_drift(drift: DriftResult) -> None:
    drift_thresh = drift.drift_alert_pct
    rows = []
    for cls_id, pct in sorted(drift.pct_change.items()):
        flagged = cls_id in drift.flagged_classes
        rows.append({
            "Class":      cls_id,
            "Δ Area (%)": f"{pct:+.1f}" if not math.isinf(pct) else "∞ (new class)",
            "Flagged":    "⚠️" if flagged else "✅",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if drift.flagged_classes:
        st.warning(
            f"**{len(drift.flagged_classes)} class(es) exceeded ±{drift_thresh:.0f}% "
            f"area change:** {drift.flagged_classes}",
            icon="⚠️",
        )
    else:
        st.success(
            f"No class exceeded the ±{drift_thresh:.0f}% drift threshold.",
            icon="✅",
        )


def _render_accuracy(acc: AccuracyResult, cfg: dict) -> None:
    min_oa = float(cfg.get("min_oa_threshold", 0.80))
    min_f1 = float(cfg.get("min_minority_f1",  0.70))

    c1, c2, c3 = st.columns(3)
    with c1:
        gate_metric("Overall Accuracy", acc.oa, min_oa,
                    fmt=".4f", higher_is_better=True)
    with c2:
        gate_metric("Cohen's Kappa",    acc.kappa, 0.6,
                    fmt=".4f", higher_is_better=True)
    with c3:
        st.metric("Valid points", acc.n_valid,
                  delta=f"{acc.n_discarded} discarded")

    # Per-class metrics
    st.markdown("**Per-class metrics**")
    rows = []
    for cls, m in acc.per_class_metrics.items():
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

    # Discard breakdown
    if acc.n_discarded > 0:
        with st.expander(f"Discarded points breakdown ({acc.n_discarded} total)"):
            discard_rows = [
                {"Reason": r.replace("_", " ").title(), "Count": c}
                for r, c in acc.discard_reasons.items() if c > 0
            ]
            st.dataframe(
                pd.DataFrame(discard_rows),
                use_container_width=True, hide_index=True,
            )


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("🗺️ Post-processing & Validation")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return
    if not session.is_unlocked("postprocess"):
        st.warning(
            "🔒 **Post-processing is locked.**  "
            "Complete **ML Classification** first.",
            icon="⚠️",
        )
        return

    classified_path_str: str | None = session.get("classified")
    if not classified_path_str or not Path(classified_path_str).exists():
        st.warning(
            "No classified raster found. Return to **ML Classification**.",
            icon="⚠️",
        )
        return

    classified_path = Path(classified_path_str)
    cfg     = session.get("config")
    out_dir = run_output_dir(cfg, run_id, "postprocessing")

    st.caption(f"Input classified raster: `{classified_path.name}`")

    # Read pixel resolution from raster
    try:
        meta        = get_meta(classified_path)
        pixel_res_m = (
            float(abs(meta["transform"].a))
            if (meta.get("crs") and meta["crs"].is_projected)
            else None
        )
    except Exception as exc:
        st.error(f"Cannot read raster metadata: {exc}", icon="❌")
        return

    if pixel_res_m is None:
        pixel_res_m_input = st.number_input(
            "Pixel resolution (m) — required for geographic CRS",
            min_value=0.1, value=10.0, step=0.5,
            key="pp_pixel_res",
        )
        pixel_res_m = float(pixel_res_m_input)

    # Tracks the "current working raster" through optional filter steps
    working_path = classified_path

    # ── A: Sieve filter (MMU) ─────────────────────────────────────────────
    st.subheader("A — Sieve Filter (Minimum Mapping Unit)")
    st.caption(
        "Removes spatial patches smaller than the MMU by absorbing them into "
        "adjacent classes. Uses GDAL SieveFilter — loads the full raster band "
        "into GDAL-managed memory."
    )

    col_mmu, col_conn = st.columns(2)
    with col_mmu:
        mmu_ha = st.number_input(
            "MMU (ha)",
            min_value=0.001, max_value=100.0,
            value=float(cfg.get("min_mapping_unit_ha", 0.5)),
            step=0.01, format="%.3f",
            key="pp_mmu_ha",
        )
    with col_conn:
        connectivity = st.radio(
            "Connectivity",
            options=[4, 8],
            format_func=lambda x: f"{x}-connected",
            horizontal=True,
            key="pp_connectivity",
        )

    sieve_out = out_dir / f"{run_id}_sieved.tif"

    if st.button("▶ Run Sieve Filter", key="run_sieve"):
        try:
            with st.spinner("Running GDAL sieve filter…"):
                _, thresh_px = sieve_filter(
                    working_path, sieve_out,
                    mmu_ha       = float(mmu_ha),
                    pixel_res_m  = pixel_res_m,
                    connectivity = int(connectivity),
                )
            working_path = sieve_out
            session.set_("classified", str(sieve_out))
            audit.log_event(
                run_id, "gate",
                {"stage": "sieve_filter", "mmu_ha": mmu_ha,
                 "threshold_px": thresh_px, "output": str(sieve_out)},
                decision="proceed",
            )
            st.success(
                f"Sieve complete — MMU threshold: **{thresh_px} px** "
                f"({mmu_ha:.3f} ha)  → `{sieve_out.name}`",
                icon="✅",
            )
        except RuntimeError as exc:
            st.error(str(exc), icon="❌")
        except Exception as exc:
            st.error(f"Sieve filter failed: {exc}", icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "sieve_filter", "error": str(exc)})
    elif sieve_out.exists():
        working_path = sieve_out
        st.caption(f"Using existing sieve output: `{sieve_out.name}`")

    # ── B: Morphological close ────────────────────────────────────────────
    st.divider()
    st.subheader("B — Morphological Closing")
    st.caption(
        "Fills small holes and smooths class boundaries using a majority-vote "
        "neighbourhood filter (dilate → erode). Processing is windowed with "
        "overlap padding — no full raster allocation in Python."
    )

    col_k, col_en = st.columns(2)
    with col_k:
        kernel_size = st.select_slider(
            "Kernel size (px, must be odd ≥ 3)",
            options=[3, 5, 7, 9],
            value=3,
            key="pp_kernel",
        )
    with col_en:
        run_morpho = st.checkbox(
            "Enable morphological closing",
            value=False,
            key="pp_morpho_enable",
        )

    morpho_out = out_dir / f"{run_id}_closed.tif"

    if run_morpho and st.button("▶ Run Morphological Closing", key="run_morpho"):
        try:
            with st.spinner("Applying morphological closing…"):
                morphological_close(
                    working_path, morpho_out,
                    kernel_size = int(kernel_size),
                )
            working_path = morpho_out
            session.set_("classified", str(morpho_out))
            audit.log_event(
                run_id, "gate",
                {"stage": "morphological_close",
                 "kernel_size": kernel_size,
                 "output": str(morpho_out)},
                decision="proceed",
            )
            st.success(
                f"Morphological close complete (k={kernel_size}) → "
                f"`{morpho_out.name}`",
                icon="✅",
            )
        except ValueError as exc:
            st.error(str(exc), icon="❌")
        except Exception as exc:
            st.error(f"Morphological close failed: {exc}", icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "morphological_close", "error": str(exc)})
    elif morpho_out.exists() and run_morpho:
        working_path = morpho_out
        st.caption(f"Using existing morphological output: `{morpho_out.name}`")

    # ── C: Class areas ────────────────────────────────────────────────────
    st.divider()
    st.subheader("C — Class Area Statistics")

    if st.button("📐 Compute Class Areas", key="compute_areas"):
        try:
            with st.spinner("Computing class areas…"):
                areas = compute_class_areas(
                    working_path,
                    nodata      = int(meta.get("nodata", -1) or -1),
                    pixel_res_m = pixel_res_m,
                )
            session.set_("class_areas", areas)
            audit.log_event(
                run_id, "gate",
                {"stage": "class_areas",
                 "n_classes": len(areas.class_ids),
                 "total_ha":  round(areas.total_area_ha, 4)},
                decision="proceed",
            )
        except Exception as exc:
            st.error(f"Area computation failed: {exc}", icon="❌")
            audit.log_event(run_id, "error",
                            {"stage": "class_areas", "error": str(exc)})

    areas: ClassAreaResult | None = session.get("class_areas")
    if areas is not None:
        df_areas = _areas_dataframe(areas)
        st.dataframe(df_areas, use_container_width=True, hide_index=True)
        st.caption(
            f"Total mapped area: **{areas.total_area_ha:.2f} ha**  |  "
            f"{len(areas.class_ids)} class(es)  |  "
            f"GSD: {areas.pixel_res_m:.1f} m"
        )

    # ── D: Drift detection ────────────────────────────────────────────────
    st.divider()
    st.subheader("D — Temporal Drift Detection")

    prev_areas_dict: dict | None = session.get("previous_class_areas")

    if areas is not None:
        col_save, col_clear = st.columns(2)
        with col_save:
            if st.button("💾 Save as Reference (next run comparison)",
                         key="save_ref_areas"):
                # Store only the lightweight areas_ha dict, not the full object
                session.set_(
                    "previous_class_areas",
                    {k: v for k, v in areas.areas_ha.items()},
                )
                audit.log_event(
                    run_id, "decision",
                    {"action": "save_reference_areas",
                     "classes": areas.class_ids,
                     "total_ha": round(areas.total_area_ha, 4)},
                    decision="proceed",
                )
                st.success("Reference areas saved for drift comparison.", icon="💾")
                st.rerun()
        with col_clear:
            if prev_areas_dict and st.button("🗑 Clear reference", key="clear_ref"):
                session.set_("previous_class_areas", None)
                st.rerun()

    if areas is None:
        st.info("Compute class areas above to enable drift detection.", icon="ℹ️")
    elif not prev_areas_dict:
        st.info(
            "No reference run saved. Save the current areas as reference, "
            "then re-run classification to detect drift.",
            icon="ℹ️",
        )
    else:
        # Reconstruct a minimal ClassAreaResult from the stored dict for check_drift
        prev_total = sum(prev_areas_dict.values())
        prev_px    = {k: int(v * 10_000 / (pixel_res_m ** 2))
                      for k, v in prev_areas_dict.items()}
        from pipeline.postprocess import ClassAreaResult as _CAR
        prev_car = _CAR(
            class_ids     = sorted(prev_areas_dict.keys()),
            pixel_counts  = prev_px,
            areas_ha      = prev_areas_dict,
            total_area_ha = prev_total,
            pixel_res_m   = pixel_res_m,
        )

        drift = check_drift(areas, prev_car, cfg=cfg)
        _render_drift(drift)

    # ── E: Accuracy assessment ────────────────────────────────────────────
    st.divider()
    st.subheader("E — Accuracy Assessment (Ground Truth CSV)")
    st.caption(
        "Upload a CSV with WGS-84 lat/lon reference points. "
        "Points are reprojected to the raster CRS and sampled from the "
        "classified raster.  Requires pyproj."
    )

    csv_upload = st.file_uploader(
        "Reference CSV (.csv)", type=["csv"], key="acc_csv_upload"
    )

    col_lat, col_lon, col_cls = st.columns(3)
    with col_lat:
        lat_col = st.text_input("Latitude column",  value="lat",   key="acc_lat")
    with col_lon:
        lon_col = st.text_input("Longitude column", value="lon",   key="acc_lon")
    with col_cls:
        cls_col = st.text_input("Class column",     value="class", key="acc_cls")

    if csv_upload is not None:
        upload_dir = run_output_dir(cfg, run_id, "uploads")
        csv_path   = save_upload(csv_upload, upload_dir)

        if st.button("▶ Run Accuracy Assessment", key="run_acc"):
            try:
                with st.spinner("Sampling reference points…"):
                    acc = assess_accuracy_from_points(
                        classified_path = working_path,
                        ref_csv_path    = csv_path,
                        lat_col         = lat_col,
                        lon_col         = lon_col,
                        class_col       = cls_col,
                        nodata          = int(meta.get("nodata", -1) or -1),
                    )
                session.set_("accuracy", acc)
                audit.log_event(
                    run_id, "gate",
                    {
                        "stage":       "accuracy_assessment",
                        "oa":          round(acc.oa, 4),
                        "kappa":       round(acc.kappa, 4),
                        "n_points":    acc.n_points,
                        "n_valid":     acc.n_valid,
                        "n_discarded": acc.n_discarded,
                    },
                    decision="proceed",
                )
            except ImportError as exc:
                st.error(str(exc), icon="❌")
            except ValueError as exc:
                st.error(str(exc), icon="❌")
                audit.log_event(run_id, "error",
                                {"stage": "accuracy_assessment",
                                 "error": str(exc)})
            except Exception as exc:
                st.error(f"Accuracy assessment failed: {exc}", icon="❌")
                audit.log_event(run_id, "error",
                                {"stage": "accuracy_assessment",
                                 "error": str(exc)})

    acc_result: AccuracyResult | None = session.get("accuracy")
    if acc_result is not None:
        _render_accuracy(acc_result, cfg)

    # ── F: Unlock gate ────────────────────────────────────────────────────
    st.divider()
    if session.is_unlocked("export"):
        st.success("Export & Delivery already unlocked.", icon="✅")
        return

    # Spatial filters are the gate — accuracy assessment is optional
    classified_final = session.get("classified")
    if not classified_final or not Path(classified_final).exists():
        st.info(
            "Complete at least one spatial filter step to unlock Export.",
            icon="ℹ️",
        )
        return

    st.success(
        "Post-processing complete. Proceed to Export & Delivery.", icon="✅"
    )
    if st.button("🔓 Unlock Export & Delivery →", type="primary"):
        session.unlock_stage("export")
        audit.log_event(
            run_id, "decision",
            {"action": "unlock_export",
             "classified_path": classified_final,
             "has_accuracy": acc_result is not None},
            decision="proceed",
        )
        st.rerun()
