"""
Page 6 — Export & Delivery.

Converts the classified raster to COG, generates a STAC 1.0 item JSON,
copies the run audit log, packages everything into a ZIP archive with
SHA-256 checksums, and provides a one-click download button.

Session state written
---------------------
session["export_manifest"] : ExportManifest — paths, checksums, sizes

Memory discipline
-----------------
write_cog, build_stac_item, and package_run all work on on-disk files.
The only data held in Python are:
  - The STAC item dict (a small JSON-serialisable dict, never pixel data).
  - The ZIP bytes fed to st.download_button — Streamlit reads these once
    from disk at button-render time; the Path object is passed, not a
    pre-loaded bytes object.
  - ExportManifest (a frozen dataclass of paths, checksums, and sizes —
    no pixel data).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import audit, session
from pipeline.export import (
    ExportManifest,
    build_stac_item,
    export_audit_log,
    package_run,
    write_cog,
    write_stac_item,
)
from pipeline.report import generate_report
from ui._helpers import run_output_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _manifest_dataframe(manifest: ExportManifest) -> pd.DataFrame:
    """One row per artifact: filename, size (KB), SHA-256 (truncated for display)."""
    rows = []
    for filename, checksum in manifest.file_checksums.items():
        size_bytes = manifest.file_sizes.get(filename, 0)
        rows.append({
            "File":          filename,
            "Size (KB)":     round(size_bytes / 1024, 1),
            "SHA-256 (hex)": checksum,
        })
    return pd.DataFrame(rows)


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Page ──────────────────────────────────────────────────────────────────────

def render() -> None:
    st.title("📦 Export & Delivery")

    run_id = session.get("run_id")
    if not run_id:
        st.info("Start a **New Run** from the sidebar to begin.", icon="👈")
        return
    if not session.is_unlocked("export"):
        st.warning(
            "🔒 **Export is locked.**  "
            "Complete **Post-processing & Validation** first.",
            icon="⚠️",
        )
        return

    classified_path_str: str | None = session.get("classified")
    if not classified_path_str or not Path(classified_path_str).exists():
        st.warning(
            "No classified raster found. Return to **Post-processing**.",
            icon="⚠️",
        )
        return

    classified_path = Path(classified_path_str)
    cfg     = session.get("config")
    out_dir = run_output_dir(cfg, run_id, "export")

    st.caption(
        f"Source raster: `{classified_path.name}`  |  "
        f"Output directory: `{out_dir}`"
    )

    # ── A: Optional STAC metadata ─────────────────────────────────────────
    st.subheader("A — STAC Metadata (optional)")
    st.caption(
        "These fields are embedded in the STAC item JSON. "
        "Leave blank to use defaults."
    )

    col_plat, col_dt = st.columns(2)
    with col_plat:
        platform = st.text_input(
            "Platform / sensor",
            value="",
            placeholder="e.g. Sentinel-1, WorldView-3",
            key="stac_platform",
        )
    with col_dt:
        stac_datetime = st.text_input(
            "Acquisition datetime (ISO 8601)",
            value="",
            placeholder=f"e.g. {_iso_now()}",
            key="stac_datetime",
        )

    extra_props: dict = {}
    if platform.strip():
        extra_props["platform"] = platform.strip()
    if stac_datetime.strip():
        extra_props["datetime"] = stac_datetime.strip()

    feat_result = session.get("features")
    if feat_result is not None:
        extra_props["terraforge:features"] = feat_result.feature_names

    cls_result = session.get("model")
    if cls_result is not None:
        extra_props["terraforge:model_type"]  = cls_result.model_type
        extra_props["terraforge:oa"]          = round(cls_result.oa, 4)
        extra_props["terraforge:minority_f1"] = round(cls_result.minority_f1, 4)
        extra_props["terraforge:kappa"]       = round(cls_result.kappa, 4)
        extra_props["classification:classes"] = cls_result.class_labels

    acc_result = session.get("accuracy")
    if acc_result is not None:
        extra_props["terraforge:accuracy_oa"]    = round(acc_result.oa, 4)
        extra_props["terraforge:accuracy_kappa"] = round(acc_result.kappa, 4)
        extra_props["terraforge:n_ref_points"]   = acc_result.n_valid

    # ── B: Build artifacts ────────────────────────────────────────────────
    st.divider()
    st.subheader("B — Build Export Artifacts")

    cog_path       = out_dir / f"{run_id}_classified_cog.tif"
    stac_path      = out_dir / f"{run_id}_stac_item.json"
    audit_log_path = out_dir / f"{run_id}_audit_log.json"

    if st.button("▶ Build COG + STAC + Audit Log", type="primary",
                 key="run_export_build"):
        build_errors: list[str] = []

        # Step 1 — write COG
        with st.spinner("Converting to Cloud Optimized GeoTIFF…"):
            try:
                write_cog(
                    src_path  = classified_path,
                    out_path  = cog_path,
                    resampling = "nearest",   # classification raster
                )
                st.success(f"COG written → `{cog_path.name}`", icon="✅")
            except Exception as exc:
                build_errors.append(f"COG: {exc}")
                st.error(f"COG failed: {exc}", icon="❌")

        # Step 2 — build + write STAC item (no pixel I/O)
        if cog_path.exists():
            with st.spinner("Generating STAC item JSON…"):
                try:
                    item = build_stac_item(
                        cog_path   = cog_path,
                        run_id     = run_id,
                        properties = extra_props or None,
                    )
                    write_stac_item(item, stac_path)
                    # Release the dict — it was only needed for serialisation
                    del item
                    st.success(f"STAC item written → `{stac_path.name}`", icon="✅")
                except Exception as exc:
                    build_errors.append(f"STAC: {exc}")
                    st.error(f"STAC item failed: {exc}", icon="❌")

        # Step 3 — export audit log (reads from logs/ via audit.get_log)
        with st.spinner("Exporting audit log…"):
            try:
                export_audit_log(run_id, audit_log_path)
                st.success(
                    f"Audit log exported → `{audit_log_path.name}`", icon="✅"
                )
            except Exception as exc:
                build_errors.append(f"Audit log: {exc}")
                st.error(f"Audit log export failed: {exc}", icon="❌")

        if build_errors:
            audit.log_event(
                run_id, "error",
                {"stage": "export_build", "errors": build_errors},
            )

    # ── C: PDF Report ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("C — Reporte PDF Técnico")
    st.caption(
        "Genera un reporte técnico corporativo en PDF con métricas de clasificación, "
        "preprocesamiento y log de auditoría completo."
    )

    pdf_path = out_dir / f"{run_id}_report.pdf"
    operator_name = st.text_input(
        "Nombre del operador",
        placeholder="Ej. Juan García — Geólogo Senior",
        key="export_operator_name",
        help="Se incluye en la portada del reporte. Puede dejarse en blanco.",
    )

    if st.button("📄 Generar Reporte PDF", key="run_pdf_report"):
        with st.spinner("Generando reporte PDF…"):
            try:
                # Build a snapshot of session data — primitives and dataclasses only
                session_snapshot = {
                    "model":    session.get("model"),
                    "accuracy": session.get("accuracy"),
                    "areas":    session.get("areas"),
                    "config":   session.get("config"),
                }
                generate_report(
                    run_id        = run_id,
                    session_data  = session_snapshot,
                    out_path      = pdf_path,
                    operator_name = operator_name.strip(),
                )
                st.success(f"Reporte generado → `{pdf_path.name}`", icon="✅")
                audit.log_event(
                    run_id, "gate",
                    {
                        "stage":    "report_generation",
                        "operator": operator_name.strip() or "—",
                        "output":   str(pdf_path),
                    },
                    decision="proceed",
                )
            except Exception as exc:
                st.error(f"Error generando el reporte: {exc}", icon="❌")
                audit.log_event(
                    run_id, "error",
                    {"stage": "report_generation", "error": str(exc)},
                )

    if pdf_path.exists():
        st.download_button(
            label     = f"⬇️  Descargar `{pdf_path.name}`",
            data      = pdf_path.read_bytes(),
            file_name = pdf_path.name,
            mime      = "application/pdf",
            key       = "download_pdf_report",
        )

    # ── D: Package into ZIP ───────────────────────────────────────────────
    st.divider()
    st.subheader("D — Package ZIP Archive")

    # Collect all artifacts that actually exist on disk (PDF included if generated)
    candidate_artifacts = [
        p for p in [cog_path, stac_path, audit_log_path, pdf_path]
        if p.exists()
    ]

    # Optionally include the feature stack if it was produced this run
    feat_result = session.get("features")
    if feat_result is not None:
        feat_path = feat_result.feature_path
        if feat_path.exists():
            candidate_artifacts.append(feat_path)

    if not candidate_artifacts:
        st.info(
            "Build the export artifacts above before packaging.", icon="ℹ️"
        )
    else:
        st.caption(
            f"Artifacts to package: "
            + ", ".join(f"`{p.name}`" for p in candidate_artifacts)
        )

        if st.button("📦 Package ZIP", key="run_package"):
            with st.spinner("Computing checksums and packaging…"):
                try:
                    manifest = package_run(
                        run_id    = run_id,
                        artifacts = candidate_artifacts,
                        out_dir   = out_dir,
                    )
                    session.set_("export_manifest", manifest)

                    audit.log_event(
                        run_id, "gate",
                        {
                            "stage":    "export_package",
                            "zip":      str(manifest.zip_path),
                            "files":    list(manifest.file_checksums.keys()),
                            "checksums": manifest.file_checksums,
                            "sizes_kb": {
                                k: round(v / 1024, 1)
                                for k, v in manifest.file_sizes.items()
                            },
                        },
                        decision="proceed",
                    )
                    st.success(
                        f"ZIP packaged → `{manifest.zip_path.name}`",
                        icon="✅",
                    )
                except Exception as exc:
                    st.error(f"Packaging failed: {exc}", icon="❌")
                    audit.log_event(
                        run_id, "error",
                        {"stage": "export_package", "error": str(exc)},
                    )

    # ── E: Manifest display ───────────────────────────────────────────────
    manifest: ExportManifest | None = session.get("export_manifest")
    if manifest is None:
        return

    st.divider()
    st.subheader("E — Export Manifest")

    col_rid, col_ts, col_zip = st.columns(3)
    col_rid.metric("Run ID",      manifest.run_id)
    col_ts.metric("Exported at",  manifest.exported_at[:19].replace("T", " "))
    col_zip.metric("Archive",     manifest.zip_path.name)

    st.markdown("**Artifact inventory**")
    df = _manifest_dataframe(manifest)
    st.dataframe(df, use_container_width=True, hide_index=True)

    total_kb = sum(manifest.file_sizes.values()) / 1024
    st.caption(
        f"Total uncompressed size: **{total_kb:.1f} KB**  |  "
        f"{len(manifest.file_checksums)} file(s)  |  "
        f"All checksums are SHA-256"
    )

    # ── F: Download ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("F — Download")

    if manifest.zip_path.exists():
        # Read bytes only at button render time; not held in a variable
        st.download_button(
            label     = f"⬇️  Download `{manifest.zip_path.name}`",
            data      = manifest.zip_path.read_bytes(),
            file_name = manifest.zip_path.name,
            mime      = "application/zip",
            type      = "primary",
            key       = "download_zip",
        )
        st.caption(
            "The ZIP contains the COG raster, STAC item JSON, and audit log. "
            "Verify integrity with the SHA-256 checksums shown above."
        )
    else:
        st.warning("ZIP file not found on disk.", icon="⚠️")

    # ── G: STAC item preview ──────────────────────────────────────────────
    if stac_path.exists():
        with st.expander("STAC item JSON preview"):
            st.code(stac_path.read_text(encoding="utf-8"), language="json")
