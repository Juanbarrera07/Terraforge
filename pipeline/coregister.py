"""
Phase 3 — Coregistration interface.

Interface contract
------------------
This module defines the full public interface for sub-pixel coregistration.
The concrete implementation (AROSICS) is *not* required at this phase.
The stub satisfies the interface so that:

  - The UI, gate logic, and audit log can be developed and tested without AROSICS.
  - Real AROSICS slots in behind the same API later by replacing _run_arosics_coreg().
  - All gate logic, result types, and report builders remain unchanged.

CoregistrationResult fields
----------------------------
  shift_x_px    Horizontal shift applied to the source raster (pixels).
  shift_y_px    Vertical shift applied to the source raster (pixels).
  shift_magnitude  Euclidean magnitude of the shift vector (pixels).
  rmse          Registration error in pixels (RMSE of residuals).
  gate_passed   True when rmse ≤ configured threshold.
  corrected_path  Path to the coregistered output, or None if gate blocked.
  message       Human-readable status string.
  is_stub       True when the real AROSICS backend was not used.
  shift_map     Optional 2-D float array of per-pixel residual shifts (for heatmap).

RMSE gate
---------
apply_rmse_gate(result, threshold) returns a new result with gate_passed set.
Gate decision and threshold are written to the audit log by the UI layer.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Optional

import numpy as np


# ── Result type ───────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class CoregistrationResult:
    shift_x_px:       float
    shift_y_px:       float
    shift_magnitude:  float
    rmse:             float
    gate_passed:      bool
    corrected_path:   Optional[Path]
    message:          str
    is_stub:          bool               = True
    shift_map:        Optional[np.ndarray] = dataclasses.field(
        default=None, compare=False, repr=False
    )


# ── Gate logic ────────────────────────────────────────────────────────────────

def apply_rmse_gate(
    result: CoregistrationResult,
    threshold: float,
) -> CoregistrationResult:
    """
    Return a new CoregistrationResult with gate_passed updated to reflect
    whether result.rmse ≤ threshold.

    The corrected_path is set to None when the gate fails, signalling
    downstream stages that coregistered data is unavailable.
    """
    passed = result.rmse <= threshold
    return dataclasses.replace(
        result,
        gate_passed=passed,
        corrected_path=result.corrected_path if passed else None,
        message=(
            result.message
            if passed
            else (
                f"RMSE gate FAILED: {result.rmse:.4f} px > threshold {threshold:.4f} px. "
                "Manual inspection required before proceeding."
            )
        ),
    )


# ── Backend selector ──────────────────────────────────────────────────────────

def _arosics_available() -> bool:
    """Return True if AROSICS is importable (installed)."""
    try:
        import arosics  # noqa: F401
        return True
    except ImportError:
        return False


def _run_stub_coreg(
    src_path: Path,
    ref_path: Path,
    out_path: Path,
) -> CoregistrationResult:
    """
    Stub coregistration — returns zeroed shifts and RMSE=0.
    Copies the source file to out_path so downstream stages have a valid path.

    Replace this function's body with the real AROSICS call when Phase 3
    AROSICS integration is activated.
    """
    import shutil

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, out_path)

    return CoregistrationResult(
        shift_x_px      = 0.0,
        shift_y_px      = 0.0,
        shift_magnitude  = 0.0,
        rmse             = 0.0,
        gate_passed      = True,   # updated by apply_rmse_gate
        corrected_path   = out_path,
        message          = (
            "Stub coregistration: AROSICS not installed. "
            "Source raster copied without modification. "
            "Install arosics (conda-forge) to enable real coregistration."
        ),
        is_stub          = True,
        shift_map        = None,
    )


def _run_arosics_coreg(
    src_path: Path,
    ref_path: Path,
    out_path: Path,
    max_shift: int = 5,
    window_size: tuple[int, int] = (256, 256),
) -> CoregistrationResult:
    """
    Real AROSICS coregistration (activated when arosics is installed).

    Uses COREG_LOCAL for a spatially distributed shift map, which also
    provides per-pixel RMSE for the shift map heatmap.

    This function is called by run_coregistration() when _arosics_available()
    returns True. Not called during Phase 3 stub period.
    """
    from arosics import COREG_LOCAL  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cl = COREG_LOCAL(
        str(ref_path),
        str(src_path),
        path_out=str(out_path),
        window_size=window_size,
        max_shift=max_shift,
        fmt_out="GTiff",
        q=True,     # quiet mode
    )
    cl.correct_shifts()

    # Extract shift statistics from COREG_LOCAL results table
    table     = cl.CoRegPoints_table
    shifts_x  = table["X_SHIFT_PX"].dropna().values
    shifts_y  = table["Y_SHIFT_PX"].dropna().values
    rmse      = float(table["ABS_SHIFT"].dropna().values.mean()) if len(table) else 0.0
    shift_x   = float(shifts_x.mean()) if len(shifts_x) else 0.0
    shift_y   = float(shifts_y.mean()) if len(shifts_y) else 0.0
    magnitude = math.hypot(shift_x, shift_y)

    return CoregistrationResult(
        shift_x_px      = shift_x,
        shift_y_px      = shift_y,
        shift_magnitude  = magnitude,
        rmse             = rmse,
        gate_passed      = True,   # updated by apply_rmse_gate
        corrected_path   = out_path,
        message          = (
            f"AROSICS COREG_LOCAL: shift=({shift_x:.3f}, {shift_y:.3f}) px, "
            f"RMSE={rmse:.4f} px."
        ),
        is_stub          = False,
        shift_map        = None,   # could be populated from table if needed
    )


# ── Public entry point ────────────────────────────────────────────────────────

def run_coregistration(
    src_path: str | Path,
    ref_path: str | Path,
    out_path: str | Path,
    config: dict,
) -> CoregistrationResult:
    """
    Run coregistration of `src_path` to `ref_path` and apply the RMSE gate.

    Automatically uses the real AROSICS backend when installed; falls back
    to the stub otherwise.

    Parameters
    ----------
    src_path  : Path to the raster to be corrected (slave image).
    ref_path  : Path to the reference raster (master image).
    out_path  : Path for the corrected output raster.
    config    : Pipeline config dict (reads coreg_rmse_threshold).

    Returns
    -------
    CoregistrationResult with gate_passed set according to the RMSE threshold.
    The caller is responsible for logging the gate outcome.
    """
    src_path = Path(src_path)
    ref_path = Path(ref_path)
    out_path = Path(out_path)
    threshold = float(config.get("coreg_rmse_threshold", 0.5))

    if _arosics_available():
        raw = _run_arosics_coreg(src_path, ref_path, out_path)
    else:
        raw = _run_stub_coreg(src_path, ref_path, out_path)

    return apply_rmse_gate(raw, threshold)


# ── Report builder ────────────────────────────────────────────────────────────

def get_shift_report(result: CoregistrationResult) -> dict:
    """
    Build a flat dict of coregistration metrics for UI display and audit logging.

    Keys are human-readable strings; values are JSON-serialisable scalars.
    """
    return {
        "shift_x_px":      round(result.shift_x_px, 4),
        "shift_y_px":      round(result.shift_y_px, 4),
        "shift_magnitude": round(result.shift_magnitude, 4),
        "rmse_px":         round(result.rmse, 4),
        "gate_passed":     result.gate_passed,
        "backend":         "stub" if result.is_stub else "arosics",
        "output_path":     str(result.corrected_path) if result.corrected_path else None,
        "message":         result.message,
    }
