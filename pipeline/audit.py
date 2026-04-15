"""
Audit logger.

Every gate outcome, user decision, and parameter change is written to
./logs/{run_id}.json with an ISO 8601 UTC timestamp.  The log is also
mirrored to st.session_state.audit_log for in-app display.

Event types
-----------
run_start      New run initialised.
config_snapshot Full pipeline configuration captured for reproducibility.
ingestion      File ingested successfully.
gate           Validation / quality gate evaluated (includes decision field).
decision       Explicit user choice (e.g. "override warning and proceed").
param_change   User changed a pipeline parameter.
error          Unhandled exception caught in UI layer.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOGS_DIR = Path("logs")

EventType = str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_path(run_id: str) -> Path:
    return _LOGS_DIR / f"{run_id}.json"


def _read(run_id: str) -> list[dict]:
    path = _log_path(run_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _write(run_id: str, log: list[dict]) -> None:
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log_path(run_id).write_text(
        json.dumps(log, indent=2, default=str),
        encoding="utf-8",
    )


# ── Public API ───────────────────────────────────────────────────────────────

def log_event(
    run_id: str,
    event_type: EventType,
    details: dict[str, Any],
    decision: Optional[str] = None,
) -> dict[str, Any]:
    """
    Append an event to the run log and return the entry dict.

    Parameters
    ----------
    run_id:     Active run identifier (8-char hex).
    event_type: See module docstring for valid values.
    details:    Arbitrary context — must be JSON-serialisable (use default=str).
    decision:   Gate/decision outcome: "proceed" | "block" | "override".
    """
    entry: dict[str, Any] = {
        "timestamp": _now(),
        "run_id": run_id,
        "event_type": event_type,
        "details": details,
    }
    if decision is not None:
        entry["decision"] = decision

    log = _read(run_id)
    log.append(entry)
    _write(run_id, log)
    return entry


def log_config_snapshot(run_id: str, config: dict) -> None:
    """
    Write a one-time configuration snapshot event for ``run_id``.

    If a snapshot for the run already exists, no duplicate event is written.
    """
    log = _read(run_id)
    if any(entry.get("event_type") == "config_snapshot" for entry in log):
        return

    entry: dict[str, Any] = {
        "timestamp": _now(),
        "run_id": run_id,
        "event_type": "config_snapshot",
        "details": config,
    }
    log.append(entry)
    _write(run_id, log)


def append_to_session(entry: dict[str, Any]) -> None:
    """Mirror a log entry to st.session_state.audit_log (in-app display only)."""
    try:
        import streamlit as st  # noqa: PLC0415 — lazy import to keep module testable

        if not isinstance(st.session_state.get("audit_log"), list):
            st.session_state.audit_log = []
        st.session_state.audit_log.append(entry)
    except Exception:  # pragma: no cover — Streamlit not available in tests
        pass


def get_log(run_id: str) -> list[dict]:
    """Return the complete audit log for a run."""
    return _read(run_id)
