"""
Page 2 — Preprocessing & Coregistration.

Applies DOS1 atmospheric correction to optical layers, Lee speckle filtering
to SAR layers, and AROSICS sub-pixel coregistration between a source and
reference layer.  Outputs are written under {tmp_dir}/{run_id}/preprocessing/.

Session state written
---------------------
session["preprocessed"]   : dict[str, str]              — layer_key → output path
session["coreg_results"]  : dict[str, CoregistrationResult]
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline import audit, session
from pipeline.coregister import CoregistrationResult, get_shift_report, run_coregistration
from pipeline.preprocess import dos1_atmospheric_correction, lee_speckle_filter
from ui._helpers import gate_metric, make_progress_cb, run_output_dir


# ── Internal helpers ──────────────────────────────────────────────────────────

def _layer_options(raw_data: dict) -> dict[str, str]:
    """Return {display_label: layer_key} for all ingested layers."""
    return {
        f"{key.split('__')[0].upper()}  —  {layer['meta'].get('crs_epsg', '?')}  "
        f"({layer['meta']['width']}×{layer['meta']['height']}, "
        f"{layer['meta']['count']} band{'s' if layer['meta']['count'] != 1 else ''})  "
        f"[{Path(str(layer['path'])).name}]": key
        for key, layer in raw_data.items()
    }


def _layer_type(raw_data: dict, layer_key: str) -> str:
    return raw_data[layer_key].get("layer_type", "unknown")


def _layer_path(raw_data: dict, preprocessed: dict, layer_key: str) -> Path:
    """Return preprocessed path if available, else original ingested path."""
    if preprocessed and layer_key in preprocessed:
        return Path(preprocessed[layer_key])
    return Path(str(raw_data[layer_key]["path"]))


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("⚙️ Preprocessing & Coregistration")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return
    if not session.is_unlocked("preprocessing"):
        prev = "ingestion"
        st.warning(
            f"🔒 **Preprocessing is locked.**  Complete **{prev.capitalize()}** first.",
            icon="⚠️",
        )
        return

    raw_data: dict | None = session.get("raw_data")
    if not raw_data:
        st.warning("No layers ingested yet. Return to **Data Ingestion**.", icon="⚠️")
        return

    cfg         = session.get("config")
    out_dir     = run_output_dir(cfg, run_id, "preprocessing")
    threshold   = float(cfg.get("coreg_rmse_threshold", 0.5))
    layer_opts  = _layer_options(raw_data)
    layer_labels = list(layer_opts.keys())

    preprocessed: dict = session.get("preprocessed") or {}
    coreg_results: dict = session.get("coreg_results") or {}

    # ── Section A: Optical preprocessing (DOS1) ───────────────────────────
    st.subheader("A — Optical Atmospheric Correction (DOS1)")
    st.caption(
        "Subtracts the per-band dark object value from every pixel. "
        "Apply to multispectral layers only."
    )

    optical_keys = [k for k in raw_data if _layer_type(raw_data, k) == "optical"]

    if not optical_keys:
        st.info("No optical layers found in current run.", icon="ℹ️")
    else:
        opt_labels = {
            f"{Path(str(raw_data[k]['path'])).name}  "
            f"({raw_data[k]['meta']['count']} bands)": k
            for k in optical_keys
        }
        selected_opt_label = st.selectbox(
            "Select optical layer for DOS1",
            options=list(opt_labels.keys()),
            key="dos1_layer_select",
        )
        selected_opt_key = opt_labels[selected_opt_label]

        if st.button("▶ Run DOS1 Correction", key="run_dos1"):
            src      = _layer_path(raw_data, preprocessed, selected_opt_key)
            out_path = out_dir / f"{src.stem}_dos1.tif"
            bar      = st.progress(0, text="DOS1 correction…")
            try:
                with st.spinner("Applying DOS1…"):
                    dos1_atmospheric_correction(
                        src, out_path,
                        progress=make_progress_cb(bar),
                    )
                bar.progress(1.0, text="Done.")
                preprocessed[selected_opt_key] = str(out_path)
                session.set_("preprocessed", preprocessed)
                audit.log_event(
                    run_id, "gate",
                    {"stage": "dos1", "layer": selected_opt_key,
                     "output": str(out_path)},
                    decision="proceed",
                )
                st.success(f"DOS1 complete → `{out_path.name}`", icon="✅")
            except Exception as exc:
                bar.empty()
                st.error(f"DOS1 failed: {exc}", icon="❌")
                audit.log_event(run_id, "error",
                                {"stage": "dos1", "error": str(exc)})

    # ── Section B: SAR preprocessing (Lee filter) ─────────────────────────
    st.divider()
    st.subheader("B — SAR Speckle Filtering (Lee)")
    st.caption("Applies the Lee filter to reduce multiplicative SAR speckle noise.")

    sar_keys = [k for k in raw_data if _layer_type(raw_data, k) == "sar"]

    if not sar_keys:
        st.info("No SAR layers found in current run.", icon="ℹ️")
    else:
        sar_labels = {
            f"{Path(str(raw_data[k]['path'])).name}  "
            f"({raw_data[k]['meta']['count']} bands)": k
            for k in sar_keys
        }
        selected_sar_label = st.selectbox(
            "Select SAR layer for Lee filter",
            options=list(sar_labels.keys()),
            key="lee_layer_select",
        )
        selected_sar_key = sar_labels[selected_sar_label]

        col_k, col_enl = st.columns(2)
        with col_k:
            kernel_size = st.select_slider(
                "Kernel size (px)",
                options=[3, 5, 7, 9, 11],
                value=7,
                key="lee_kernel",
                help="Larger kernel → smoother result; loses fine spatial detail.",
            )
        with col_enl:
            enl = st.number_input(
                "ENL (Equivalent Number of Looks)",
                min_value=0.5, max_value=20.0,
                value=float(cfg.get("sar_enl_default", 1.0)),
                step=0.5,
                key="lee_enl",
                help="Higher ENL → less aggressive filtering. "
                     "1 = single-look (conservative).",
            )

        if st.button("▶ Run Lee Filter", key="run_lee"):
            src      = _layer_path(raw_data, preprocessed, selected_sar_key)
            out_path = out_dir / f"{src.stem}_lee{kernel_size}.tif"
            bar      = st.progress(0, text="Lee speckle filter…")
            try:
                with st.spinner("Applying Lee filter…"):
                    lee_speckle_filter(
                        src, out_path,
                        kernel_size=int(kernel_size),
                        enl=float(enl),
                        progress=make_progress_cb(bar),
                    )
                bar.progress(1.0, text="Done.")
                preprocessed[selected_sar_key] = str(out_path)
                session.set_("preprocessed", preprocessed)
                audit.log_event(
                    run_id, "gate",
                    {"stage": "lee_filter", "layer": selected_sar_key,
                     "kernel": kernel_size, "enl": enl,
                     "output": str(out_path)},
                    decision="proceed",
                )
                st.success(f"Lee filter complete → `{out_path.name}`", icon="✅")
            except Exception as exc:
                bar.empty()
                st.error(f"Lee filter failed: {exc}", icon="❌")
                audit.log_event(run_id, "error",
                                {"stage": "lee_filter", "error": str(exc)})

    # ── Section C: Coregistration ─────────────────────────────────────────
    st.divider()
    st.subheader("C — Coregistration (AROSICS)")
    st.caption(
        "Align the source raster to the reference using sub-pixel shift correction. "
        "RMSE gate threshold: "
        f"**{threshold} px** (from `pipeline_config.yaml`)."
    )

    if len(layer_labels) < 2:
        st.info(
            "At least two ingested layers are required for coregistration.",
            icon="ℹ️",
        )
    else:
        col_src, col_ref = st.columns(2)
        with col_src:
            src_label = st.selectbox(
                "Source (image to correct)",
                options=layer_labels,
                key="coreg_src",
            )
        with col_ref:
            ref_options = [l for l in layer_labels if l != src_label]
            ref_label   = st.selectbox(
                "Reference (anchor image)",
                options=ref_options,
                key="coreg_ref",
            )

        src_key = layer_opts[src_label]
        ref_key = layer_opts[ref_label]

        if st.button("▶ Run Coregistration", type="primary", key="run_coreg"):
            src_path = _layer_path(raw_data, preprocessed, src_key)
            ref_path = _layer_path(raw_data, preprocessed, ref_key)
            out_path = out_dir / f"{src_path.stem}_coreg.tif"

            with st.spinner("Running coregistration…"):
                try:
                    result = run_coregistration(src_path, ref_path, out_path, cfg)
                    coreg_results[src_key] = result
                    session.set_("coreg_results", coreg_results)

                    # Update preprocessed path to coregistered output
                    if result.corrected_path is not None:
                        preprocessed[src_key] = str(result.corrected_path)
                        session.set_("preprocessed", preprocessed)

                    report = get_shift_report(result)
                    audit.log_event(
                        run_id, "gate",
                        {"stage": "coregistration", "layer": src_key,
                         **report},
                        decision="proceed" if result.gate_passed else "block",
                    )
                except Exception as exc:
                    st.error(f"Coregistration failed: {exc}", icon="❌")
                    audit.log_event(run_id, "error",
                                    {"stage": "coregistration", "error": str(exc)})

    # ── Section D: Gate display ───────────────────────────────────────────
    if coreg_results:
        st.divider()
        st.subheader("D — Coregistration Results")

        for layer_key, result in coreg_results.items():
            layer_name = layer_key.split("__")[-1]
            with st.expander(f"**{layer_name}**", expanded=True):
                if result.is_stub:
                    st.info(
                        "⚠️ AROSICS not installed — stub coregistration used. "
                        "Source raster copied unchanged. "
                        "Install `arosics` (conda-forge) for real coregistration.",
                        icon="🔧",
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    gate_metric(
                        "RMSE", result.rmse, threshold,
                        fmt=".4f", higher_is_better=False,
                    )
                with c2:
                    st.metric("Shift X (px)", f"{result.shift_x_px:.4f}")
                with c3:
                    st.metric("Shift Y (px)", f"{result.shift_y_px:.4f}")
                with c4:
                    st.metric("Magnitude (px)", f"{result.shift_magnitude:.4f}")

                if result.gate_passed:
                    st.success(result.message, icon="✅")
                else:
                    st.error(result.message, icon="❌")

    # ── Section E: Preprocessed layer inventory ───────────────────────────
    if preprocessed:
        st.divider()
        st.subheader("Preprocessed Layers")
        for key, path in preprocessed.items():
            name  = key.split("__")[-1]
            ppath = Path(path)
            icon  = "✅" if ppath.exists() else "❌"
            st.markdown(f"{icon} **{name}** → `{ppath.name}`")

    # ── Section F: Unlock gate ────────────────────────────────────────────
    st.divider()
    any_coreg_passed = any(r.gate_passed for r in coreg_results.values()) \
        if coreg_results else False
    any_preprocessed = bool(preprocessed)

    if session.is_unlocked("features"):
        st.success("Feature Engineering already unlocked.", icon="✅")
        return

    if not any_preprocessed:
        st.info(
            "Run at least one preprocessing step to unlock Feature Engineering.",
            icon="ℹ️",
        )
        return

    if coreg_results and not any_coreg_passed:
        st.error(
            f"**Coregistration RMSE gate failed** (RMSE > {threshold} px).  "
            "Inspect the source data or override below.",
            icon="🚫",
        )
        override = st.checkbox(
            "⚠️ Override RMSE gate and proceed anyway",
            key="coreg_override",
        )
        if not override:
            return
        audit.log_event(
            run_id, "decision",
            {"action": "override_coreg_rmse_gate",
             "threshold_px": threshold},
            decision="override",
        )

    if st.button("🔓 Unlock Feature Engineering →", type="primary"):
        session.unlock_stage("features")
        audit.log_event(
            run_id, "decision",
            {"action": "unlock_features",
             "preprocessed_layers": list(preprocessed.keys()),
             "coreg_gate_passed": any_coreg_passed},
            decision="proceed",
        )
        st.success("Feature Engineering unlocked!")
        st.rerun()
