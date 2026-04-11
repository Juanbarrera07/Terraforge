"""
Page 1 — Data Ingestion.

Supports two ingestion modes:

  Upload mode   — drag-and-drop files through the browser; suitable for files
                  small enough to transit a network connection.
  Local path    — paste an absolute path on the server / local filesystem;
                  the correct choice for large rasters (multi-GB GeoTIFFs,
                  SAR scenes) that must not travel through the browser.

In both modes the ingestion core is path-based: every file lands on disk
before ``ingest_path()`` is called.  No UploadedFile objects ever enter the
pipeline directly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import audit, session
from pipeline.ingest import build_metadata_table, ingest_path
from pipeline.validate import (
    ValidationResult,
    has_critical_failures,
    run_all_validations,
    validation_summary,
)
from ui._helpers import run_upload_dir, save_upload

_STATUS_ICONS = {"ok": "✅", "warn": "⚠️", "error": "❌"}

_VALID_SUFFIXES = {".tif", ".tiff", ".zip"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_validation_result(r: ValidationResult) -> None:
    label = r.check.replace("_", " ").title()
    msg   = f"**{label}** — {r.message}"
    if r.status == "ok":
        st.success(msg, icon=_STATUS_ICONS["ok"])
    elif r.status == "warn":
        st.warning(msg, icon=_STATUS_ICONS["warn"])
    else:
        st.error(msg, icon=_STATUS_ICONS["error"])


def _validate_local_path(raw: str, label: str) -> Path | None:
    """
    Validate a user-supplied path string.  Returns a resolved Path on success,
    or shows an st.error and returns None on failure.
    """
    raw = raw.strip()
    if not raw:
        return None  # empty → skip (field is optional)

    p = Path(raw)

    if not p.exists():
        st.error(f"**{label}**: file not found — `{p}`", icon="❌")
        return None
    if not p.is_file():
        st.error(f"**{label}**: path is not a regular file — `{p}`", icon="❌")
        return None
    if p.suffix.lower() not in _VALID_SUFFIXES:
        st.error(
            f"**{label}**: unsupported extension `{p.suffix}`. "
            f"Expected one of: {', '.join(sorted(_VALID_SUFFIXES))}",
            icon="❌",
        )
        return None
    try:
        p.open("rb").close()
    except PermissionError:
        st.error(f"**{label}**: permission denied — `{p}`", icon="❌")
        return None

    return p


# ── Upload mode ───────────────────────────────────────────────────────────────

def _upload_mode(tmp_dir: Path) -> list[tuple[str, Path]]:
    """
    Render file-uploader widgets and return (hint, disk_path) pairs for every
    file the user has attached.  Files are saved to disk immediately.
    """
    st.caption(
        "Upload SAR, multispectral, and/or DEM files. "
        "ZIP archives containing .tif files are supported for SAR.  "
        "For files larger than a few hundred MB, use **Local file path** mode."
    )

    col_sar, col_opt, col_dem = st.columns(3)
    with col_sar:
        sar_uploads = st.file_uploader(
            "SAR (.tif / .zip)", type=["tif", "tiff", "zip"],
            accept_multiple_files=True, key="upload_sar",
        )
    with col_opt:
        opt_uploads = st.file_uploader(
            "Multispectral (.tif)", type=["tif", "tiff"],
            accept_multiple_files=True, key="upload_optical",
        )
    with col_dem:
        dem_uploads = st.file_uploader(
            "DEM (.tif)", type=["tif", "tiff"],
            accept_multiple_files=True, key="upload_dem",
        )

    pairs: list[tuple[str, Path]] = []
    for hint, uploads in (
        ("sar", sar_uploads or []),
        ("optical", opt_uploads or []),
        ("dem", dem_uploads or []),
    ):
        for uf in uploads:
            disk_path = save_upload(uf, tmp_dir)
            pairs.append((hint, disk_path))

    return pairs


# ── Local path mode ───────────────────────────────────────────────────────────

def _local_path_mode() -> list[tuple[str, Path]]:
    """
    Render text inputs for absolute server-side paths and return validated
    (hint, path) pairs.  Does not touch upload state.
    """
    st.caption(
        "Enter absolute paths to files already present on the server "
        "or local filesystem.  Suitable for large rasters that cannot "
        "be uploaded through the browser."
    )

    col_sar, col_opt, col_dem = st.columns(3)
    with col_sar:
        sar_raw = st.text_input(
            "SAR path (.tif / .zip)",
            placeholder="/data/scenes/S1A_IW_20230101.tif",
            key="local_sar",
        )
    with col_opt:
        opt_raw = st.text_input(
            "Optical path (.tif)",
            placeholder="/data/scenes/S2B_MSIL2A_20230101.tif",
            key="local_optical",
        )
    with col_dem:
        dem_raw = st.text_input(
            "DEM path (.tif)",
            placeholder="/data/dem/srtm_30m.tif",
            key="local_dem",
        )

    pairs: list[tuple[str, Path]] = []
    for hint, raw in (("sar", sar_raw), ("optical", opt_raw), ("dem", dem_raw)):
        p = _validate_local_path(raw, hint.upper())
        if p is not None:
            pairs.append((hint, p))

    return pairs


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.title("📥 Data Ingestion")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return

    cfg     = session.get("config")
    tmp_dir = run_upload_dir(cfg, run_id)

    # ── Mode selector ──────────────────────────────────────────────────────
    st.subheader("Add imagery")
    mode = st.radio(
        "Ingestion mode",
        options=["Upload file", "Local file path"],
        horizontal=True,
        help=(
            "**Upload file** — drag files from your computer (best for files < 500 MB).  "
            "**Local file path** — paste a path on the server; no size limit."
        ),
    )

    st.write("")
    if mode == "Upload file":
        file_pairs = _upload_mode(tmp_dir)
    else:
        file_pairs = _local_path_mode()

    if not file_pairs:
        st.info("Provide at least one raster file to begin.", icon="📂")
        return

    # ── Ingest + Validate ──────────────────────────────────────────────────
    if st.button("🔍 Ingest & Validate", type="primary"):
        raw_data: dict[str, dict] = {}
        errors:   list[str]       = []

        with st.spinner("Ingesting files…"):
            for hint, disk_path in file_pairs:
                try:
                    # Core is path-based — no UploadedFile enters the pipeline
                    layer = ingest_path(disk_path, extract_dir=tmp_dir)
                    if hint == "dem":
                        layer["layer_type"] = "dem"
                    layer_key = f"{hint}__{disk_path.name}"
                    raw_data[layer_key] = layer

                    entry = audit.log_event(
                        run_id, "ingestion",
                        {
                            "file":   disk_path.name,
                            "sensor": layer["sensor"],
                            "type":   layer["layer_type"],
                            "crs":    str(layer["meta"].get("crs_epsg")),
                            "bands":  layer["meta"]["count"],
                        },
                    )
                    audit.append_to_session(entry)

                except Exception as exc:
                    errors.append(f"`{disk_path.name}`: {exc}")
                    entry = audit.log_event(
                        run_id, "error",
                        {"file": disk_path.name, "error": str(exc)},
                    )
                    audit.append_to_session(entry)

        for err in errors:
            st.error(f"Ingestion error — {err}", icon="❌")

        if raw_data:
            session.set_("raw_data", raw_data)

            with st.spinner("Running validation checks…"):
                v_results = run_all_validations(raw_data, cfg)
            session.set_("validation_results", v_results)

            summary     = validation_summary(v_results)
            critical_ok = not has_critical_failures(v_results)

            entry = audit.log_event(
                run_id, "gate",
                {
                    "gate":    "ingestion_validation",
                    "summary": summary,
                    "checks":  {k: v.status for k, v in v_results.items()},
                },
                decision="proceed" if critical_ok else "block",
            )
            audit.append_to_session(entry)

    # ── Layer metadata table ───────────────────────────────────────────────
    raw_data = session.get("raw_data")
    if raw_data:
        st.divider()
        st.subheader("Layer metadata")
        df = pd.DataFrame(build_metadata_table(raw_data))
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Validation results ─────────────────────────────────────────────────
    v_results: dict[str, ValidationResult] | None = session.get("validation_results")
    if v_results:
        st.divider()
        st.subheader("Validation checklist")

        summary = validation_summary(v_results)
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Passed",   summary.get("ok",   0))
        c2.metric("⚠️ Warnings", summary.get("warn", 0))
        c3.metric("❌ Errors",   summary.get("error", 0))
        st.write("")
        for result in v_results.values():
            _render_validation_result(result)

        # ── Unlock gate ────────────────────────────────────────────────────
        st.divider()
        if has_critical_failures(v_results):
            st.error(
                "**Critical validation failures detected.**  "
                "Resolve all ❌ errors before proceeding.",
                icon="🚫",
            )
        else:
            if session.is_unlocked("preprocessing"):
                st.success("Preprocessing already unlocked.", icon="✅")
            else:
                st.success("All critical checks passed. Ready to proceed.", icon="✅")
                if st.button("🔓 Unlock Preprocessing →", type="primary"):
                    session.unlock_stage("preprocessing")
                    entry = audit.log_event(
                        run_id, "decision",
                        {"action": "unlock_preprocessing",
                         "warnings": summary.get("warn", 0)},
                        decision="proceed",
                    )
                    audit.append_to_session(entry)
                    st.success("Preprocessing stage unlocked!")
                    st.rerun()
