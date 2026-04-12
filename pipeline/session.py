"""
Streamlit session state schema + helpers.

All session state keys are defined here as the single source of truth.
Use get() / set_() to access state rather than indexing st.session_state
directly — this avoids magic-string bugs across modules.

Pipeline stages (in order)
--------------------------
ingestion → preprocessing → features → classification → postprocess → export

A stage is "unlocked" once the previous stage completes all critical gates.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

# Ordered list of pipeline stages
PIPELINE_STAGES: list[str] = [
    "ingestion",
    "preprocessing",
    "features",
    "classification",
    "postprocess",
    "export",
]

# Default values for every session key.
# None means "not yet populated"; callers must check before use.
_DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "raw_data": None,               # dict[str, LayerDict]  — from ingest.py
    "validation_results": None,     # dict[str, ValidationResult] — from validate.py
    "preprocessed": None,           # dict  — after preprocess.py
    "features": None,               # dict  — after features.py
    "model": None,                  # trained model object
    "classified": None,             # classified raster path
    "accuracy": None,               # accuracy metrics dict
    "audit_log": None,              # list[dict] — mirrored from logs/
    "config": None,                 # loaded pipeline config dict
    "pipeline_unlocked": None,      # set[str] of unlocked stage names
    "previous_class_areas": None,   # dict — for drift monitor
    "coreg_results":        None,   # dict[str, CoregistrationResult] — from preprocessing
    "class_areas":          None,   # ClassAreaResult — from postprocess
    "export_manifest":      None,   # ExportManifest — from export
    # ── Training data metadata ────────────────────────────────────────────────
    "training_source":      None,   # str  — "label_raster" | "shapefile"
    "training_path":        None,   # str  — path to shapefile or label raster
    "class_column":         None,   # str  — class column name (shapefile only)
    # ── Post-processing chain async state ────────────────────────────────────
    "chain_thread":         None,   # threading.Thread | None
    "chain_queue":          None,   # queue.Queue | None
    "chain_log":            None,   # list[str] — accumulated progress messages
    "chain_start_time":     None,   # float — time.time() at chain start
    "chain_cancel_event":   None,   # threading.Event | None
}


def init_session(config: dict) -> None:
    """
    Idempotent initialiser — only sets keys that are absent from session state.
    Call once at the top of app.py on every Streamlit run.
    """
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default

    if st.session_state.config is None:
        st.session_state.config = config

    if st.session_state.pipeline_unlocked is None:
        st.session_state.pipeline_unlocked = {"ingestion"}


def _cleanup_run_tmp(run_id: str, tmp_dir: str | Path) -> None:
    """
    Remove ``tmp/{run_id}/`` for a completed or abandoned run.

    - Best-effort: logs on error, never raises.
    - Security: verifies the target is a direct child of *tmp_dir* before
      calling shutil.rmtree, so a malformed run_id cannot escape the sandbox.
    - Does NOT touch logs/ or outputs/ — only the tmp subtree.

    Parameters
    ----------
    run_id  : The run identifier string (8-char hex, uppercase).
    tmp_dir : Root tmp directory from config (e.g. ``"tmp"`` or ``Path("tmp")``).
    """
    if not run_id:
        return

    tmp_root = Path(tmp_dir).resolve()
    target = (tmp_root / run_id).resolve()

    # Security guard: target must be an immediate child of tmp_root.
    try:
        target.relative_to(tmp_root)
    except ValueError:
        print(f"[session] WARN: cleanup target {target} is outside {tmp_root} — skipped")
        return

    if not target.exists():
        return

    try:
        shutil.rmtree(target)
        print(f"[session] Cleaned tmp/{run_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"[session] WARN: could not remove {target}: {exc}")


def new_run() -> str:
    """
    Generate a new run ID and reset all pipeline-specific state.
    Cleans up the previous run's tmp directory (best-effort).
    Returns the new run_id string.
    """
    # Capture previous run context BEFORE resetting state.
    old_run_id: str | None = st.session_state.get("run_id")
    cfg: dict = st.session_state.get("config") or {}

    run_id = uuid.uuid4().hex[:8].upper()
    st.session_state.run_id = run_id
    st.session_state.raw_data = {}
    st.session_state.validation_results = {}
    st.session_state.preprocessed = None
    st.session_state.features = None
    st.session_state.model = None
    st.session_state.classified = None
    st.session_state.accuracy = None
    st.session_state.audit_log = []
    st.session_state.pipeline_unlocked = {"ingestion"}
    st.session_state.previous_class_areas = None
    st.session_state.coreg_results        = None
    st.session_state.class_areas          = None
    st.session_state.export_manifest      = None
    st.session_state.training_source      = None
    st.session_state.training_path        = None
    st.session_state.class_column         = None
    st.session_state.chain_thread         = None
    st.session_state.chain_queue          = None
    st.session_state.chain_log            = None
    st.session_state.chain_start_time     = None
    st.session_state.chain_cancel_event   = None

    # Clean previous run's tmp — best-effort, never blocks the new run.
    if old_run_id and cfg:
        _cleanup_run_tmp(old_run_id, cfg.get("tmp_dir", "tmp"))

    return run_id


def unlock_stage(stage: str) -> None:
    """Mark a pipeline stage as accessible."""
    if stage in PIPELINE_STAGES:
        unlocked: set = st.session_state.pipeline_unlocked or set()
        unlocked.add(stage)
        st.session_state.pipeline_unlocked = unlocked


def is_unlocked(stage: str) -> bool:
    """Return True if the given stage has been unlocked."""
    unlocked = st.session_state.get("pipeline_unlocked") or set()
    return stage in unlocked


def get(key: str, default: Any = None) -> Any:
    """Safe session state getter with an explicit default."""
    return st.session_state.get(key, default)


def set_(key: str, value: Any) -> None:
    """Set a session state key."""
    st.session_state[key] = value
