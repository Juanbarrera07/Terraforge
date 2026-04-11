"""
TerraForge Mining Intelligence — Streamlit entry point.

This file is a thin router only.  All page logic lives in ui/.
Navigation: sidebar radio → 6 pages.
Pages 2–6 are locked until their prerequisite stage is completed.
Session state is initialised once per session via pipeline.session.init_session().
"""
from __future__ import annotations

import streamlit as st

from pipeline import audit, session
from pipeline.config_loader import load_config

# Page modules
from ui import (
    page_ingestion,
    page_preprocessing,
    page_features,
    page_classification,
    page_postprocess,
    page_export,
)

# ── App config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TerraForge Mining Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Bootstrap ─────────────────────────────────────────────────────────────────

config = load_config()
session.init_session(config)

# ── Sidebar ───────────────────────────────────────────────────────────────────

PAGES: list[str] = [
    "📥 Data Ingestion",
    "⚙️ Preprocessing & Coregistration",
    "🔬 Feature Engineering",
    "🤖 ML Classification",
    "🗺️ Post-processing & Validation",
    "📦 Export & Delivery",
]

with st.sidebar:
    st.title("⛏️ TerraForge")
    st.caption("Mining Intelligence Platform")
    st.divider()

    selected_page = st.radio("Navigation", PAGES, label_visibility="collapsed")

    st.divider()

    # ── Run management ─────────────────────────────────────────────────────
    run_id: str | None = session.get("run_id")
    if run_id:
        st.success(f"Run **{run_id}**", icon="🆔")
    else:
        st.info("No active run — start one below.", icon="ℹ️")

    if st.button("🆕 New Run", use_container_width=True, type="primary"):
        rid   = session.new_run()
        entry = audit.log_event(rid, "run_start", {"action": "new_run_via_button"})
        audit.append_to_session(entry)
        st.rerun()

    # ── Pipeline stage status ──────────────────────────────────────────────
    st.divider()
    st.caption("Pipeline stages")
    unlocked: set = session.get("pipeline_unlocked") or {"ingestion"}

    _STAGE_ICONS = {
        "ingestion":      "📥",
        "preprocessing":  "⚙️",
        "features":       "🔬",
        "classification": "🤖",
        "postprocess":    "🗺️",
        "export":         "📦",
    }
    for stage, icon in _STAGE_ICONS.items():
        label = stage.replace("postprocess", "post-process").capitalize()
        if stage in unlocked:
            st.markdown(f"✅ {icon} {label}")
        else:
            st.markdown(f"🔒 {icon} {label}")


# ── Router ────────────────────────────────────────────────────────────────────

_PAGE_FUNCS = {
    PAGES[0]: page_ingestion.render,
    PAGES[1]: page_preprocessing.render,
    PAGES[2]: page_features.render,
    PAGES[3]: page_classification.render,
    PAGES[4]: page_postprocess.render,
    PAGES[5]: page_export.render,
}

_PAGE_FUNCS[selected_page]()
