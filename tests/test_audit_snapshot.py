"""Tests for audit config snapshot logging."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline import audit as audit_mod


def test_log_config_snapshot_writes_event(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_logs = tmp_path / "logs"
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    run_id = "SNAP0001"
    config = {"min_overlap_pct": 80.0, "tmp_dir": "tmp"}

    audit_mod.log_config_snapshot(run_id, config)

    events = audit_mod.get_log(run_id)
    assert len(events) == 1
    assert events[0]["event_type"] == "config_snapshot"
    assert events[0]["run_id"] == run_id
    datetime.fromisoformat(events[0]["timestamp"])


def test_log_config_snapshot_is_idempotent_for_same_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_logs = tmp_path / "logs"
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    run_id = "SNAP0002"
    config = {"coreg_rmse_threshold": 0.5}

    audit_mod.log_config_snapshot(run_id, config)
    audit_mod.log_config_snapshot(run_id, config)

    events = audit_mod.get_log(run_id)
    assert len(events) == 1
    assert [event["event_type"] for event in events] == ["config_snapshot"]


def test_log_config_snapshot_stores_config_verbatim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_logs = tmp_path / "logs"
    monkeypatch.setattr(audit_mod, "_LOGS_DIR", fake_logs)

    run_id = "SNAP0003"
    config = {
        "thresholds": {"oa": 0.8, "minority_f1": 0.7},
        "tmp_dir": "tmp",
        "flags": ["sar", "drone"],
    }

    audit_mod.log_config_snapshot(run_id, config)

    events = audit_mod.get_log(run_id)
    assert events[0]["details"] == config
