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
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import numpy as np
import rasterio

from pipeline import audit, session
from pipeline.postprocess import (
    AccuracyResult,
    ClassAreaResult,
    DriftResult,
    QualityGateResult,
    assess_accuracy_from_points,
    build_chain_step_plan,
    check_drift,
    compute_class_areas,
    confidence_filter,
    drone_adaptive_params,
    estimate_chain_time,
    has_gate_failures,
    median_smooth,
    morphological_close,
    run_postprocess_chain,
    run_quality_gates,
    sieve_filter,
)
from pipeline.raster_io import get_meta, iter_windows
from ui._helpers import gate_metric, run_output_dir, save_upload

# Fallback step names for satellite mode (overridden dynamically)
_CHAIN_STEPS_DEFAULT: list[tuple[str, str]] = [
    ("confidence_filter",    "Confidence filter"),
    ("median_smooth",        "Median smooth"),
    ("morphological_close",  "Morphological close"),
    ("sieve_filter",         "Sieve MMU filter"),
]


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


def _confidence_stats(confidence_path: Path, threshold: float) -> dict:
    """
    Windowed pass over the confidence raster to gather min/max/mean and the
    fraction of valid pixels below the threshold.  No full-raster read.
    """
    total_sum   = 0.0
    total_count = 0
    below_count = 0
    global_min  = float("inf")
    global_max  = float("-inf")

    with rasterio.open(confidence_path) as ds:
        for win in iter_windows(ds):
            tile = ds.read(1, window=win).astype(np.float32)
            valid = tile >= 0.0   # confidence nodata is -9999.0
            if not valid.any():
                continue
            vals = tile[valid]
            total_sum   += float(vals.sum())
            total_count += int(valid.sum())
            below_count += int((vals < threshold).sum())
            global_min   = min(global_min, float(vals.min()))
            global_max   = max(global_max, float(vals.max()))

    mean = total_sum / total_count if total_count > 0 else 0.0
    pct_below = below_count / total_count * 100.0 if total_count > 0 else 0.0
    return {
        "min":         global_min if total_count > 0 else float("nan"),
        "max":         global_max if total_count > 0 else float("nan"),
        "mean":        mean,
        "n_valid":     total_count,
        "n_below":     below_count,
        "pct_below":   pct_below,
    }


def _render_quality_gates(gate_results: list[QualityGateResult]) -> None:
    """Display quality gate results as a color-coded summary table."""
    _ICONS = {"pass": "✅", "warning": "⚠️", "fail": "❌"}
    rows = []
    for r in gate_results:
        icon = _ICONS.get(r.status, "")
        rows.append({
            "Metric":          r.metric_name,
            "Value":           f"{r.value:.4f}",
            "Pass threshold":  f"{r.threshold_pass}",
            "Fail threshold":  f"{r.threshold_fail}",
            "Result":          f"{icon} {r.status.upper()}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    n_fail = sum(1 for r in gate_results if r.status == "fail")
    n_warn = sum(1 for r in gate_results if r.status == "warning")
    n_pass = sum(1 for r in gate_results if r.status == "pass")
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ PASS",    n_pass)
    c2.metric("⚠️ WARNING", n_warn)
    c3.metric("❌ FAIL",    n_fail)

    for r in gate_results:
        if r.status == "fail":
            st.error(r.message, icon="❌")
        elif r.status == "warning":
            st.warning(r.message, icon="⚠️")


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

    # ── Auto: Post-Processing Chain ───────────────────────────────────────
    st.subheader("▶ Post-Processing Chain (Express)")
    st.caption(
        "Runs the post-processing chain in sequence. **Satellite mode** uses "
        "4 fixed steps; **drone mode** adapts the number and size of smoothing "
        "passes to the pixel resolution. Parameters are read from the pipeline config."
    )

    confidence_path_str: str | None = session.get("confidence")

    if not confidence_path_str or not Path(confidence_path_str).exists():
        st.info(
            "Confidence raster not available. "
            "Re-run **ML Classification** to generate it alongside the "
            "classified raster.",
            icon="ℹ️",
        )
    else:
        confidence_path = Path(confidence_path_str)
        conf_threshold  = float(cfg.get("confidence_threshold", 0.6))

        # ── Confidence map statistics (windowed, no full read) ────────────
        # Cached in session state — not recomputed on every polling rerun.
        _cstats_cached = session.get("_confidence_stats_cache")
        _cstats_key    = str(confidence_path)
        if (
            _cstats_cached is None
            or _cstats_cached.get("_path_key") != _cstats_key
        ):
            try:
                _cstats_val = _confidence_stats(confidence_path, conf_threshold)
                _cstats_val["_path_key"] = _cstats_key
                session.set_("_confidence_stats_cache", _cstats_val)
                _cstats_cached = _cstats_val
            except Exception as exc:
                _cstats_cached = None
                st.warning(f"Could not compute confidence statistics: {exc}",
                           icon="⚠️")

        with st.expander("📊 Confidence Map Statistics", expanded=True):
            if _cstats_cached:
                cstats = _cstats_cached
                cs1, cs2, cs3, cs4 = st.columns(4)
                with cs1:
                    st.metric("Mean confidence", f"{cstats['mean']:.3f}")
                with cs2:
                    st.metric("Min / Max",
                              f"{cstats['min']:.3f} / {cstats['max']:.3f}")
                with cs3:
                    st.metric("Pixels below threshold",
                              f"{cstats['n_below']:,}",
                              help=f"Confidence threshold: {conf_threshold}")
                with cs4:
                    st.metric("% below threshold", f"{cstats['pct_below']:.1f}%")

        chain_out_dir = run_output_dir(cfg, run_id, "postprocessing")

        # ── Chain configuration panel ─────────────────────────────────────
        with st.expander("⚙️ Chain Configuration", expanded=False):
            st.caption("Parameters are read from `pipeline_config.yaml`.")
            _cfg_params = {
                "confidence_threshold":     cfg.get("confidence_threshold", 0.6),
                "median_filter_size":       cfg.get("median_filter_size", 3),
                "morphological_kernel_size":cfg.get("morphological_kernel_size", 3),
                "morphological_iterations": cfg.get("morphological_iterations", 1),
                "min_mapping_unit_ha":      cfg.get("min_mapping_unit_ha", 0.5),
            }
            drone_threshold = float(cfg.get("drone_pixel_res_threshold_m", 1.0))
            is_drone = pixel_res_m is not None and pixel_res_m < drone_threshold

            _param_rows = [{"Parameter": k, "Value": str(v)}
                           for k, v in _cfg_params.items()]
            st.dataframe(pd.DataFrame(_param_rows),
                         use_container_width=True, hide_index=True)

            if is_drone:
                st.info(
                    f"**Drone mode** will be auto-enabled "
                    f"(pixel_res_m = {pixel_res_m:.3f} m < {drone_threshold} m). "
                    "Parameters are adaptively computed for high-resolution imagery.",
                    icon="🛸",
                )
                # Show adaptive parameters computed for this resolution
                try:
                    _dp = drone_adaptive_params(pixel_res_m, cfg)
                    st.markdown("**Adaptive smoothing plan**")
                    _sp_rows = []
                    for _sp in _dp["smooth_passes"]:
                        _sp_rows.append({
                            "Pass":      _sp["label"],
                            "Kernel":    f"{_sp['kernel']}×{_sp['kernel']}",
                            "Target":    f"{_sp['target_m']:.1f} m",
                            "Method":    _sp["method"],
                        })
                    st.dataframe(pd.DataFrame(_sp_rows),
                                 use_container_width=True, hide_index=True)
                    _ad_rows = [
                        {"Parameter": "Morpho kernel",    "Value": str(_dp["morpho_kernel"])},
                        {"Parameter": "Morpho iterations", "Value": str(_dp["morpho_iters"])},
                        {"Parameter": "Conf filter size",  "Value": str(_dp["conf_filter_size"])},
                        {"Parameter": "Sieve min px",      "Value": str(_dp["sieve_min_px"])},
                        {"Parameter": "Skip confidence",   "Value": str(_dp["skip_confidence"])},
                    ]
                    st.dataframe(pd.DataFrame(_ad_rows),
                                 use_container_width=True, hide_index=True)
                except Exception:
                    pass  # best-effort display
            else:
                st.caption(
                    f"pixel_res_m = {pixel_res_m:.2f} m — "
                    f"standard mode (drone mode threshold: {drone_threshold} m)"
                )

            # Time estimate table (best-effort)
            try:
                _res_for_est = pixel_res_m if pixel_res_m else 10.0
                _estimates   = estimate_chain_time(classified_path, cfg, _res_for_est)
                _warn_msg    = _estimates.pop("warning", None)
                _dyn_steps   = build_chain_step_plan(_res_for_est, cfg)
                _est_rows = [
                    {"Step": slabel,
                     "Estimated time": f"~{_estimates[skey]:.0f}s"}
                    for skey, slabel in _dyn_steps
                    if skey in _estimates
                ]
                _num_vals = {v for v in _estimates.values()
                             if isinstance(v, (int, float))}
                _total_s = sum(_num_vals)
                _est_rows.append({"Step": "**TOTAL**",
                                  "Estimated time": f"**~{_total_s/60:.1f} min**"})
                st.markdown("**Estimated processing time**")
                st.dataframe(pd.DataFrame(_est_rows),
                             use_container_width=True, hide_index=True)
                if _warn_msg:
                    st.warning(f"⏱️ {_warn_msg}", icon="⏱️")
            except Exception:
                pass  # estimates are best-effort

        # ── Thread state helpers ──────────────────────────────────────────
        _chain_thread: threading.Thread | None = session.get("chain_thread")
        _chain_queue:  queue.Queue | None      = session.get("chain_queue")
        _chain_running = _chain_thread is not None and _chain_thread.is_alive()

        # ── Launch button (only if not running) ───────────────────────────
        if not _chain_running:
            # Build dynamic step plan based on pixel resolution
            _dynamic_steps = build_chain_step_plan(pixel_res_m, cfg)

            if st.button("▶ Run Post-Processing Chain", key="run_chain",
                         type="primary"):
                _q  = queue.Queue()
                _ev = threading.Event()

                # Capture values so the thread closure gets correct references.
                _pixel_res_capture = pixel_res_m
                _step_plan_capture = list(_dynamic_steps)

                def _run_chain_in_thread(  # redefine to inject pixel_res_m
                    classified_path=classified_path,
                    confidence_path=confidence_path,
                    cfg=cfg,
                    out_dir=chain_out_dir,
                    run_id=run_id,
                    q=_q,
                    cancel_event=_ev,
                    pixel_res_m=_pixel_res_capture,
                    step_plan=_step_plan_capture,
                ) -> None:
                    """Execute chain in background thread; communicate via queue."""
                    _step_keys    = [s for s, _ in step_plan]
                    _current_step = [0]

                    def _progress(msg: str) -> None:
                        step_idx = next(
                            (i + 1 for i, s in enumerate(_step_keys) if s in msg),
                            _current_step[0] + 1,
                        )
                        _current_step[0] = step_idx
                        ts = datetime.now().strftime("%H:%M:%S")
                        q.put({
                            "step":  step_idx,
                            "total": len(_step_keys),
                            "msg":   msg,
                            "done":  False,
                            "error": None,
                            "log":   f"[{ts}] Step {step_idx}/{len(_step_keys)}: {msg}",
                        })

                    try:
                        result = run_postprocess_chain(
                            classified_path = classified_path,
                            confidence_path = confidence_path,
                            cfg             = cfg,
                            out_dir         = out_dir,
                            run_id          = run_id,
                            progress        = _progress,
                            cancel_event    = cancel_event,
                            pixel_res_m     = pixel_res_m,
                        )
                        ts = datetime.now().strftime("%H:%M:%S")
                        q.put({
                            "step":   len(_step_keys),
                            "total":  len(_step_keys),
                            "msg":    "Chain complete",
                            "done":   True,
                            "error":  None,
                            "result": result,
                            "log":    f"[{ts}] Chain complete — {result['final'].name}",
                        })
                    except Exception as exc:  # noqa: BLE001
                        ts = datetime.now().strftime("%H:%M:%S")
                        q.put({
                            "step":  0,
                            "total": len(_step_keys),
                            "msg":   str(exc),
                            "done":  True,
                            "error": str(exc),
                            "log":   f"[{ts}] ERROR: {exc}",
                        })

                _t = threading.Thread(target=_run_chain_in_thread, daemon=True)
                session.set_("chain_thread",       _t)
                session.set_("chain_queue",        _q)
                session.set_("chain_cancel_event", _ev)
                session.set_("chain_log",          [])
                session.set_("chain_start_time",   time.time())
                session.set_("_chain_step_plan",   _step_plan_capture)
                _t.start()
                st.rerun()

        # ── Polling loop (runs while thread is alive) ─────────────────────
        _chain_thread = session.get("chain_thread")
        _chain_queue  = session.get("chain_queue")
        _chain_running = _chain_thread is not None and _chain_thread.is_alive()

        if _chain_running or (
            _chain_thread is not None
            and not _chain_thread.is_alive()
            and _chain_queue is not None
            and not _chain_queue.empty()
        ):
            _start_t  = session.get("chain_start_time") or time.time()
            _elapsed  = time.time() - _start_t
            _log_msgs: list[str] = session.get("chain_log") or []
            _active_steps: list[tuple[str, str]] = (
                session.get("_chain_step_plan") or _CHAIN_STEPS_DEFAULT
            )
            _last_step  = 0
            _last_total = len(_active_steps)
            _done_msg   = None

            # Drain the queue
            while True:
                try:
                    _msg = _chain_queue.get_nowait()
                    _log_msgs.append(_msg["log"])
                    _last_step  = _msg["step"]
                    _last_total = _msg["total"]
                    if _msg["done"]:
                        _done_msg = _msg
                        break
                except queue.Empty:
                    break

            session.set_("chain_log", _log_msgs)

            # Render status panel
            _frac = min(1.0, _last_step / _last_total) if _last_total else 0.0
            _step_label = (
                _active_steps[_last_step - 1][1]
                if 0 < _last_step <= len(_active_steps)
                else "Starting…"
            )
            col_prog, col_time = st.columns([3, 1])
            with col_prog:
                st.progress(_frac,
                            text=f"Step {_last_step}/{_last_total}: {_step_label}")
            with col_time:
                st.metric("Elapsed", f"{_elapsed:.0f}s")

            # Step checklist (dynamic — adapts to drone/satellite mode)
            for i, (_, slabel) in enumerate(_active_steps):
                if i < _last_step:
                    icon = "✅"
                elif i == _last_step:
                    icon = "🔄"
                else:
                    icon = "⏳"
                st.markdown(f"{icon} **{i+1}. {slabel}**")

            # Log expander
            with st.expander("📋 Processing Log", expanded=_chain_running):
                log_text = "\n".join(_log_msgs) if _log_msgs else "Waiting for first step…"
                st.code(log_text, language=None)

            # Cancel button
            _cancel_ev = session.get("chain_cancel_event")
            if _chain_running and _cancel_ev is not None:
                if st.button("⏹ Cancel", key="cancel_chain"):
                    _cancel_ev.set()
                    session.set_("chain_log",
                                 _log_msgs + ["[user] Cancellation requested…"])

            # Handle completion
            if _done_msg is not None:
                session.set_("chain_thread",       None)
                session.set_("chain_queue",        None)
                session.set_("chain_cancel_event", None)

                if _done_msg["error"]:
                    err = _done_msg["error"]
                    if "Cancelled" in err:
                        st.warning("Chain cancelled by user.", icon="⏹")
                    else:
                        st.error(f"Post-processing chain failed: {err}", icon="❌")
                        audit.log_event(
                            run_id, "error",
                            {"stage": "postprocess_chain", "error": err},
                        )
                else:
                    chain_result = _done_msg["result"]
                    final_path   = chain_result["final"]
                    total_elapsed = time.time() - _start_t
                    session.set_("classified",    str(final_path))
                    session.set_("_chain_result", chain_result)
                    working_path = final_path
                    audit.log_event(
                        run_id, "decision",
                        {"action": "run_postprocess_chain",
                         "final_output": str(final_path),
                         "elapsed_s": round(total_elapsed, 1)},
                        decision="proceed",
                    )

                    # Result summary
                    st.markdown("---")
                    _n_steps = len(_active_steps)
                    for i, (_, slabel) in enumerate(_active_steps):
                        st.markdown(f"✅ Step {i+1}/{_n_steps}: {slabel}")
                    st.markdown("─" * 40)
                    try:
                        _sz_mb = final_path.stat().st_size / (1024 * 1024)
                        _sz_str = f" ({_sz_mb:.1f} MB)"
                    except Exception:
                        _sz_str = ""
                    st.success(
                        f"Chain complete — Total: {total_elapsed:.1f}s\n\n"
                        f"Output: `{final_path.name}`{_sz_str}",
                        icon="✅",
                    )

            elif _chain_running:
                # 0.1 s is enough to avoid busy-spin; 0.5 s caused noticeable
                # UI unresponsiveness because it blocked the Streamlit main thread.
                time.sleep(0.1)
                st.rerun()

        # ── Before / after comparison ─────────────────────────────────────
        chain_result_stored = session.get("_chain_result")
        if chain_result_stored:
            final_chain_path = chain_result_stored.get("final")
            if final_chain_path and Path(final_chain_path).exists():
                st.markdown("**Before / After Comparison**")
                bc1, bc2 = st.columns(2)

                def _quick_areas(path: Path) -> dict[int, int] | None:
                    """Return {class: pixel_count} without computing ha."""
                    counts: dict[int, int] = {}
                    try:
                        with rasterio.open(path) as ds:
                            nd = int(ds.nodata) if ds.nodata is not None else -1
                            for w in iter_windows(ds):
                                tile = ds.read(1, window=w)
                                vals, cnts = np.unique(
                                    tile[tile != nd], return_counts=True
                                )
                                for v, c in zip(vals.tolist(), cnts.tolist()):
                                    counts[v] = counts.get(v, 0) + c
                    except Exception:
                        return None
                    return counts

                with bc1:
                    st.caption("**Before** (original classified)")
                    before_counts = _quick_areas(classified_path)
                    if before_counts:
                        st.dataframe(
                            pd.DataFrame(
                                [{"Class": k, "Pixels": v}
                                 for k, v in sorted(before_counts.items())]
                            ),
                            use_container_width=True, hide_index=True,
                        )
                with bc2:
                    st.caption("**After** (chain final output)")
                    after_counts = _quick_areas(Path(final_chain_path))
                    if after_counts:
                        st.dataframe(
                            pd.DataFrame(
                                [{"Class": k, "Pixels": v}
                                 for k, v in sorted(after_counts.items())]
                            ),
                            use_container_width=True, hide_index=True,
                        )

    st.divider()

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

    # ── ▶: Quality Gate Assessment ────────────────────────────────────────
    st.divider()
    st.subheader("▶ Quality Gate Assessment")
    st.caption(
        "Evaluates classification quality across 3-tier gates "
        "(PASS / WARNING / FAIL). All thresholds are read from "
        "`pipeline_config.yaml`. Re-run after accuracy assessment to include "
        "independent OA in the evaluation."
    )

    # Build confidence_stats for the gate (only need mean)
    _conf_stats_gate: dict | None = None
    _confidence_path_str = session.get("confidence")
    if _confidence_path_str and Path(_confidence_path_str).exists():
        try:
            _cstats = _confidence_stats(
                Path(_confidence_path_str),
                float(cfg.get("confidence_threshold", 0.6)),
            )
            _conf_stats_gate = {"mean": _cstats["mean"]}
        except Exception:
            pass

    # Compute nodata fraction if class areas are available
    _nodata_pct: float | None = None
    _areas_now: ClassAreaResult | None = session.get("class_areas")
    if _areas_now is not None:
        try:
            _total_px = meta["width"] * meta["height"]
            _valid_px = sum(_areas_now.pixel_counts.values())
            if _total_px > 0:
                _nodata_pct = (_total_px - _valid_px) / _total_px
        except Exception:
            pass

    if st.button("▶ Evaluate Quality Gates", key="run_quality_gates",
                 type="primary"):
        _gate_results = run_quality_gates(
            model_result     = session.get("model"),
            accuracy_result  = session.get("accuracy"),
            class_areas      = _areas_now,
            confidence_stats = _conf_stats_gate,
            cfg              = cfg,
            nodata_pct       = _nodata_pct,
        )
        session.set_("quality_gate_results", _gate_results)
        audit.log_event(
            run_id, "gate",
            {
                "stage":   "quality_gate_assessment",
                "results": [
                    {"metric": r.metric_name,
                     "value":  round(r.value, 4),
                     "status": r.status}
                    for r in _gate_results
                ],
            },
            decision=(
                "block"   if has_gate_failures(_gate_results)
                else "proceed"
            ),
        )
        st.rerun()

    gate_results: list[QualityGateResult] | None = session.get(
        "quality_gate_results"
    )
    if gate_results is not None:
        _render_quality_gates(gate_results)

    # ── F: Unlock gate ────────────────────────────────────────────────────
    st.divider()
    if session.is_unlocked("export"):
        st.success("Export & Delivery already unlocked.", icon="✅")
        return

    # Spatial filters are the minimum requirement
    classified_final = session.get("classified")
    if not classified_final or not Path(classified_final).exists():
        st.info(
            "Complete at least one spatial filter step to unlock Export.",
            icon="ℹ️",
        )
        return

    # Quality gate results required before unlock
    if gate_results is None:
        st.info(
            "Run **▶ Evaluate Quality Gates** above before unlocking Export.",
            icon="ℹ️",
        )
        return

    if has_gate_failures(gate_results):
        st.error(
            "**Quality gate FAILED.**  "
            "Resolve the issues above or override to proceed.",
            icon="🚫",
        )
        override = st.checkbox(
            "⚠️ Override quality gate failures and proceed anyway",
            key="gate_override_export",
        )
        if override:
            st.warning(
                "Gate override active — exported results may not meet quality "
                "standards. This decision is recorded in the audit log.",
                icon="⚠️",
            )
            if st.button("🔓 Unlock Export & Delivery (override) →"):
                session.unlock_stage("export")
                audit.log_event(
                    run_id, "decision",
                    {
                        "action":          "override_quality_gate_export",
                        "failed_metrics":  [
                            r.metric_name for r in gate_results
                            if r.status == "fail"
                        ],
                        "classified_path": classified_final,
                    },
                    decision="override",
                )
                st.rerun()
    else:
        _n_warn = sum(1 for r in gate_results if r.status == "warning")
        if _n_warn > 0:
            st.warning(
                f"Quality gate passed with **{_n_warn} warning(s)** — "
                "review results above before exporting.",
                icon="⚠️",
            )
        else:
            st.success(
                "All quality gates passed. Proceed to Export & Delivery.",
                icon="✅",
            )
        if st.button("🔓 Unlock Export & Delivery →", type="primary"):
            session.unlock_stage("export")
            audit.log_event(
                run_id, "decision",
                {
                    "action":           "unlock_export",
                    "classified_path":  classified_final,
                    "gate_status":      "warning" if _n_warn > 0 else "pass",
                    "n_warnings":       _n_warn,
                    "has_accuracy":     acc_result is not None,
                },
                decision="proceed",
            )
            st.rerun()
