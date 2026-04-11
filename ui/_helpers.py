"""
Shared utilities for TerraForge Streamlit page modules.

All functions here are UI-layer helpers only — no pipeline logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st


ProgressCallback = Callable[[int, int], None]


def save_upload(uploaded_file, dest_dir: Path) -> Path:
    """
    Write a Streamlit UploadedFile to *dest_dir* using chunked I/O.

    Files are flushed to disk in 128 KiB chunks before any pipeline function
    is called, so no in-memory copy of the raster lingers in the Streamlit
    process.  For rasters too large for browser upload, use local path mode
    instead.

    Returns the on-disk Path.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / uploaded_file.name
    uploaded_file.seek(0)
    with open(dest, "wb") as fh:
        while True:
            chunk = uploaded_file.read(1 << 17)  # 128 KiB chunks
            if not chunk:
                break
            fh.write(chunk)
    return dest


def make_progress_cb(bar: st.delta_generator.DeltaGenerator) -> ProgressCallback:
    """
    Return a ProgressCallback that advances a ``st.progress`` bar.

    The callback signature matches all pipeline progress parameters:
    ``(current_tile: int, total_tiles: int) -> None``.
    """
    def _cb(current: int, total: int) -> None:
        bar.progress(current / max(total, 1))
    return _cb


def run_output_dir(cfg: dict, run_id: str, stage: str) -> Path:
    """
    Return and create the canonical output directory for a pipeline stage.

    Layout:  ``{tmp_dir}/{run_id}/{stage}/``
    """
    d = Path(cfg.get("tmp_dir", "tmp")) / run_id / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_upload_dir(cfg: dict, run_id: str) -> Path:
    """
    Return and create the upload staging directory for a run.

    Layout:  ``{tmp_dir}/{run_id}/uploads/``
    """
    d = Path(cfg.get("tmp_dir", "tmp")) / run_id / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def gate_metric(label: str, value: float, threshold: float,
                fmt: str = ".4f", higher_is_better: bool = True) -> None:
    """
    Render an ``st.metric`` coloured red/green relative to a threshold.

    Parameters
    ----------
    higher_is_better : True for OA/F1 (pass ≥ threshold);
                       False for RMSE (pass ≤ threshold).
    """
    passes = (value >= threshold) if higher_is_better else (value <= threshold)
    delta  = value - threshold
    # delta_color "normal" → green when positive (good); invert for RMSE
    color  = "normal" if higher_is_better else "inverse"
    st.metric(
        label     = label,
        value     = f"{value:{fmt}}",
        delta     = f"{delta:+{fmt}} vs {threshold:{fmt}} threshold",
        delta_color = color if passes else "off",
        help      = f"Gate threshold: {threshold:{fmt}}",
    )
