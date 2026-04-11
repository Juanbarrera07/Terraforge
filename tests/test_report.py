"""
Tests for pipeline/report.py — generate_report() PDF output.

These tests do NOT require a Streamlit context, a real raster, or a real
audit log.  They exercise the report generator with synthetic session_data
dictionaries built from dataclass-like SimpleNamespace objects, verifying:

  1. The function creates a non-empty PDF file.
  2. It handles None / missing session data gracefully (no exception).
  3. It produces a valid PDF with full pipeline session data.

Isolation
---------
Audit log reads are patched via monkeypatch on pipeline.audit._LOGS_DIR so no
real log files are touched or required.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from pipeline import audit as audit_mod
from pipeline.report import generate_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_out(tmp_path: Path) -> Path:
    """Return a directory under pytest's tmp_path for report output."""
    out = tmp_path / "reports"
    out.mkdir()
    return out


def _fake_model(
    oa: float = 0.91,
    kappa: float = 0.88,
    minority_f1: float = 0.79,
    model_type: str = "random_forest",
    smote_applied: bool = True,
    gate_passed: bool = True,
    gate_message: str = "Gate passed.",
    n_estimators: int = 100,
    max_depth: int | None = None,
    k_folds: int = 5,
    random_state: int = 42,
    per_class_metrics: dict | None = None,
    cv_scores: list | None = None,
    feature_importances: dict | None = None,
    class_labels: list | None = None,
):
    """Return a SimpleNamespace that mimics ClassificationResult."""
    return types.SimpleNamespace(
        oa=oa,
        kappa=kappa,
        minority_f1=minority_f1,
        model_type=model_type,
        smote_applied=smote_applied,
        gate_passed=gate_passed,
        gate_message=gate_message,
        n_estimators=n_estimators,
        max_depth=max_depth,
        k_folds=k_folds,
        random_state=random_state,
        per_class_metrics=per_class_metrics or {
            1: {"precision": 0.92, "recall": 0.90, "f1": 0.91, "support": 200},
            2: {"precision": 0.88, "recall": 0.85, "f1": 0.86, "support": 120},
        },
        cv_scores=cv_scores or [0.89, 0.91, 0.90, 0.92, 0.88],
        feature_importances=feature_importances or {"ndvi": 0.35, "sar_vv": 0.25},
        class_labels=class_labels or [1, 2],
    )


def _fake_accuracy(
    oa: float = 0.89,
    kappa: float = 0.85,
    n_points: int = 400,
    n_valid: int = 390,
    n_discarded: int = 10,
):
    """Return a SimpleNamespace that mimics AccuracyResult."""
    return types.SimpleNamespace(
        oa=oa,
        kappa=kappa,
        n_points=n_points,
        n_valid=n_valid,
        n_discarded=n_discarded,
        per_class_metrics={
            1: {"precision": 0.90, "recall": 0.88, "f1": 0.89, "support": 200},
            2: {"precision": 0.87, "recall": 0.83, "f1": 0.85, "support": 190},
        },
        discard_reasons={"out_of_bounds": 7, "nodata_pixel": 3},
    )


def _fake_areas(
    areas_ha: dict | None = None,
    pixel_counts: dict | None = None,
    total_area_ha: float = 1250.5,
    pixel_res_m: float = 10.0,
):
    """Return a SimpleNamespace that mimics ClassAreaResult."""
    return types.SimpleNamespace(
        areas_ha=areas_ha or {1: 800.2, 2: 450.3},
        pixel_counts=pixel_counts or {1: 8002000, 2: 4503000},
        total_area_ha=total_area_ha,
        pixel_res_m=pixel_res_m,
        class_ids=[1, 2],
    )


def _fake_config() -> dict:
    return {
        "target_crs":            "EPSG:32618",
        "target_resolution_m":   10,
        "sar_enl_default":       1.0,
        "sar_enl_sentinel1_iw":  4.9,
        "min_overlap_pct":       80.0,
        "max_resolution_ratio":  2.0,
        "max_date_gap_days":     30,
        "min_oa_threshold":      0.80,
        "min_minority_f1":       0.70,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_generate_report_creates_file(
    tmp_out: Path, monkeypatch, tmp_path: Path
) -> None:
    """
    generate_report() with minimal session_data must create a non-empty PDF.
    """
    fake_logs = tmp_path / "logs"
    fake_logs.mkdir()
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    out_pdf = tmp_out / "test_report.pdf"
    result  = generate_report(
        run_id        = "ABCD1234",
        session_data  = {},           # no pipeline data — all N/D
        out_path      = out_pdf,
        operator_name = "Test User",
    )

    assert result == out_pdf.resolve()
    assert out_pdf.exists(), "PDF file was not created"
    assert out_pdf.stat().st_size > 0, "PDF file is empty"


def test_generate_report_with_empty_accuracy(
    tmp_out: Path, monkeypatch, tmp_path: Path
) -> None:
    """
    When accuracy is None, generate_report() must not raise and must produce
    a valid PDF file — all accuracy fields render as 'N/D'.
    """
    fake_logs = tmp_path / "logs"
    fake_logs.mkdir()
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    session_data = {
        "model":    _fake_model(),
        "accuracy": None,           # explicitly absent
        "areas":    _fake_areas(),
        "config":   _fake_config(),
    }

    out_pdf = tmp_out / "no_accuracy.pdf"
    result  = generate_report(
        run_id       = "NOACC001",
        session_data = session_data,
        out_path     = out_pdf,
    )

    assert out_pdf.exists()
    assert out_pdf.stat().st_size > 0


def test_generate_report_with_full_data(
    tmp_out: Path, monkeypatch, tmp_path: Path
) -> None:
    """
    generate_report() with complete session_data (model + accuracy + areas +
    config + audit events) must produce a non-empty PDF without raising.
    """
    # Seed a real audit log so the audit section renders table rows
    fake_logs = tmp_path / "logs"
    fake_logs.mkdir()
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    run_id = "FULL9999"
    audit_mod.log_event(run_id, "gate",
                        {"stage": "classification", "oa": 0.91},
                        decision="proceed")
    audit_mod.log_event(run_id, "decision",
                        {"action": "unlock_postprocess"},
                        decision="proceed")
    audit_mod.log_event(run_id, "gate",
                        {"stage": "report_generation", "operator": "QA Bot"},
                        decision="proceed")

    session_data = {
        "model":    _fake_model(gate_passed=True),
        "accuracy": _fake_accuracy(),
        "areas":    _fake_areas(),
        "config":   _fake_config(),
    }

    out_pdf = tmp_out / "full_report.pdf"
    result  = generate_report(
        run_id        = run_id,
        session_data  = session_data,
        out_path      = out_pdf,
        operator_name = "QA Bot",
    )

    assert out_pdf.exists()
    size = out_pdf.stat().st_size
    assert size > 4096, f"PDF unexpectedly small ({size} bytes) for full data"
