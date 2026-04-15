"""
TerraForge Mining Intelligence - Streamlit entry point.

This file is a thin router only. All page logic lives in ui/.
Navigation: sidebar radio -> 6 pages.
Pages 2-6 are locked until their prerequisite stage is completed.
Session state is initialised once per session via pipeline.session.init_session().
"""
from __future__ import annotations

import streamlit as st

from pipeline import audit, session
from pipeline.config_loader import load_config
from ui import (
    page_classification,
    page_export,
    page_features,
    page_ingestion,
    page_postprocess,
    page_preprocessing,
)

st.set_page_config(
    page_title="TerraForge Mining Intelligence",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _ensure_config_snapshot(config: dict) -> None:
    """Persist the active config exactly once for the current run."""
    run_id = session.get("run_id")
    if not run_id:
        return

    if session.get("config_snapshot_logged_run_id") == run_id:
        return

    audit.log_config_snapshot(run_id, config)
    session.set_("config_snapshot_logged_run_id", run_id)


config = load_config()
session.init_session(config)
_ensure_config_snapshot(config)

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

    run_id: str | None = session.get("run_id")
    if run_id:
        st.success(f"Run **{run_id}**", icon="🆔")
    else:
        st.info("No active run - start one below.", icon="ℹ️")

    if st.button("🆕 New Run", use_container_width=True, type="primary"):
        rid = session.new_run()
        audit.log_config_snapshot(rid, config)
        session.set_("config_snapshot_logged_run_id", rid)
        entry = audit.log_event(rid, "run_start", {"action": "new_run_via_button"})
        audit.append_to_session(entry)
        st.rerun()

    st.divider()
    st.caption("Pipeline stages")
    unlocked: set = session.get("pipeline_unlocked") or {"ingestion"}

    stage_icons = {
        "ingestion": "📥",
        "preprocessing": "⚙️",
        "features": "🔬",
        "classification": "🤖",
        "postprocess": "🗺️",
        "export": "📦",
    }
    for stage, icon in stage_icons.items():
        label = stage.replace("postprocess", "post-process").capitalize()
        if stage in unlocked:
            st.markdown(f"✅ {icon} {label}")
        else:
            st.markdown(f"🔒 {icon} {label}")


page_funcs = {
    PAGES[0]: page_ingestion.render,
    PAGES[1]: page_preprocessing.render,
    PAGES[2]: page_features.render,
    PAGES[3]: page_classification.render,
    PAGES[4]: page_postprocess.render,
    PAGES[5]: page_export.render,
}

page_funcs[selected_page]()
