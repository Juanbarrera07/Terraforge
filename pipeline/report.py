"""
pipeline/report.py — Technical PDF report generator for TerraForge.

Produces a multi-section corporate PDF from pipeline session data using
reportlab.  All work is disk-based: no pixel arrays are held in memory.

Public API
----------
generate_report(run_id, session_data, out_path, operator_name="") -> Path
compute_confidence_stats(confidence_path, threshold) -> dict | None

Session data keys consumed (all optional — missing keys yield "N/A")
---------------------------------------------------------------------
"model"             : ClassificationResult  — OA, Kappa, F1, model type, SMOTE, CV
"accuracy"          : AccuracyResult        — independent accuracy assessment
"class_areas"       : ClassAreaResult       — per-class hectare areas
"config"            : dict                  — pipeline_config.yaml values
"confidence_stats"  : dict                  — pre-computed confidence raster stats
"postprocess_chain" : dict                  — chain step results from run_postprocess_chain

Design rules
------------
- No Streamlit imports.
- Every dict/attr access goes through _g() — the report never raises on
  missing or None session data.
- Audit log rows are capped at MAX_AUDIT_ROWS to keep file size predictable.
- Confidence percentiles use reservoir sampling (O(sample_size) memory).
"""
from __future__ import annotations

import random
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── reportlab imports ─────────────────────────────────────────────────────────
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ── Corporate palette ─────────────────────────────────────────────────────────
_NAVY  = colors.HexColor("#1B2A4A")
_GRAY  = colors.HexColor("#6B7280")
_GREEN = colors.HexColor("#16A34A")
_RED   = colors.HexColor("#DC2626")
_AMBER = colors.HexColor("#D97706")
_LIGHT = colors.HexColor("#F1F5F9")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AUDIT_ROWS  = 50
_SAMPLE_SIZE    = 100_000   # reservoir size for percentile estimation
PAGE_W, PAGE_H  = A4
MARGIN          = 2.0 * cm


# ── Safe accessor ─────────────────────────────────────────────────────────────

def _g(obj: Any, *keys, default: str = "N/A") -> str:
    """
    Safely navigate nested dicts/objects returning a display string.

    Usage:
        _g(session_data, "model", "oa")          # dict path
        _g(result_obj, "oa")                      # attribute access
    """
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    if cur is None:
        return default
    if isinstance(cur, float):
        return f"{cur:.4f}"
    if isinstance(cur, bool):
        return "Yes" if cur else "No"
    return str(cur)


# ── Confidence stats (public — importable by page_export) ─────────────────────

def compute_confidence_stats(confidence_path: Path, threshold: float) -> dict | None:
    """
    Windowed pass over a confidence raster using reservoir sampling to estimate
    P05/P50/P95 without loading the full raster.  O(_SAMPLE_SIZE) memory.

    Returns a dict with keys: p05, p50, p95, mean, min, max, n_valid,
    n_above, pct_above, threshold.  Returns None on any error.
    """
    try:
        import numpy as np
        import rasterio
        from pipeline.raster_io import iter_windows

        reservoir: list[float] = []
        total_sum   = 0.0
        total_count = 0
        n_above     = 0
        global_min  = float("inf")
        global_max  = float("-inf")

        with rasterio.open(confidence_path) as ds:
            for win in iter_windows(ds):
                tile  = ds.read(1, window=win).astype(float)
                valid = tile >= 0.0
                if not valid.any():
                    continue
                vals = tile[valid]
                total_sum   += float(vals.sum())
                total_count += int(valid.sum())
                n_above     += int((vals >= threshold).sum())
                global_min   = min(global_min, float(vals.min()))
                global_max   = max(global_max, float(vals.max()))

                # Reservoir sampling for percentiles
                for v in vals.tolist():
                    if len(reservoir) < _SAMPLE_SIZE:
                        reservoir.append(v)
                    else:
                        j = random.randint(0, total_count - 1)
                        if j < _SAMPLE_SIZE:
                            reservoir[j] = v

        if total_count == 0:
            return None

        arr = np.array(reservoir)
        mean     = total_sum / total_count
        pct_above = n_above / total_count * 100.0

        return {
            "p05":       float(np.percentile(arr, 5)),
            "p50":       float(np.percentile(arr, 50)),
            "p95":       float(np.percentile(arr, 95)),
            "mean":      mean,
            "min":       global_min,
            "max":       global_max,
            "n_valid":   total_count,
            "n_above":   n_above,
            "pct_above": pct_above,
            "threshold": threshold,
        }
    except Exception:
        return None


# ── Styles ────────────────────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        parent=base["Title"],
        fontSize=28,
        textColor=_NAVY,
        spaceAfter=0.4 * cm,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        parent=base["Normal"],
        fontSize=13,
        textColor=_GRAY,
        spaceAfter=0.3 * cm,
    )
    styles["section_h1"] = ParagraphStyle(
        "section_h1",
        parent=base["Heading1"],
        fontSize=14,
        textColor=_NAVY,
        spaceBefore=0.6 * cm,
        spaceAfter=0.25 * cm,
        borderPad=2,
        leading=18,
    )
    styles["section_h2"] = ParagraphStyle(
        "section_h2",
        parent=base["Heading2"],
        fontSize=11,
        textColor=_NAVY,
        spaceBefore=0.4 * cm,
        spaceAfter=0.15 * cm,
    )
    styles["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.black,
        spaceAfter=0.2 * cm,
        leading=13,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontSize=8,
        textColor=_GRAY,
        spaceAfter=0.15 * cm,
        leading=11,
    )
    styles["footer"] = ParagraphStyle(
        "footer",
        parent=base["Normal"],
        fontSize=7,
        textColor=_GRAY,
    )
    return styles


# ── Footer / page-number callback ─────────────────────────────────────────────

def _make_footer_cb(run_id: str, generated_at: str):
    """Return an onPage callback that draws the footer on every page."""

    def _draw_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(_GRAY)
        footer_text = (
            f"TerraForge Mining Intelligence  |  Run ID: {run_id}  |  "
            f"Generated: {generated_at}  |  Page {doc.page}"
        )
        canvas.drawString(MARGIN, 1.0 * cm, footer_text)
        # horizontal rule above footer
        canvas.setStrokeColor(_NAVY)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 1.4 * cm, PAGE_W - MARGIN, 1.4 * cm)
        canvas.restoreState()

    return _draw_footer


# ── Logo placeholder Flowable ─────────────────────────────────────────────────

class _LogoPlaceholder(Flowable):
    """Draws a navy rectangle as a placeholder for the corporate logo."""

    def __init__(self, width: float = 5.0 * cm, height: float = 1.8 * cm):
        super().__init__()
        self.width  = width
        self.height = height

    def draw(self):
        self.canv.setFillColor(_NAVY)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        self.canv.setFillColor(colors.white)
        self.canv.setFont("Helvetica-Bold", 11)
        self.canv.drawCentredString(
            self.width / 2, self.height / 2 - 4, "TerraForge"
        )


# ── Table helpers ─────────────────────────────────────────────────────────────

_HEADER_STYLE = TableStyle([
    ("BACKGROUND",  (0, 0), (-1, 0), _NAVY),
    ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
    ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",    (0, 0), (-1, -1), 8),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT, colors.white]),
    ("GRID",        (0, 0), (-1, -1), 0.3, _GRAY),
    ("TOPPADDING",  (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 5),
    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
])


def _two_col_table(pairs: list[tuple[str, str]], col_widths=None) -> Table:
    """Build a label→value table from (label, value) pairs."""
    if col_widths is None:
        col_widths = [6 * cm, 10 * cm]
    data = [["Parameter", "Value"]] + list(pairs)
    t = Table(data, colWidths=col_widths)
    t.setStyle(_HEADER_STYLE)
    return t


def _gate_badge(status: str) -> str:
    """Return a coloured text label for a gate status string."""
    return {"PASS": "PASS ✓", "WARNING": "WARNING ⚠", "FAIL": "FAIL ✗"}.get(
        status.upper(), status
    )


# ── Section builders ──────────────────────────────────────────────────────────

def _section_cover(
    styles: dict,
    run_id: str,
    operator_name: str,
    generated_at: str,
    story: list,
) -> None:
    story.append(Spacer(1, 2.5 * cm))
    story.append(_LogoPlaceholder())
    story.append(Spacer(1, 1.2 * cm))
    story.append(Paragraph("Technical Classification Report", styles["cover_title"]))
    story.append(Paragraph("TerraForge Mining Intelligence", styles["cover_sub"]))
    story.append(Spacer(1, 0.8 * cm))

    cover_data = [
        ("Run ID",     run_id),
        ("Operator",   operator_name or "—"),
        ("Date/Time",  generated_at),
    ]
    t = Table(
        [[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
         for k, v in cover_data],
        colWidths=[5 * cm, 11 * cm],
    )
    t.setStyle(TableStyle([
        ("FONTNAME",  (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, _NAVY),
    ]))
    story.append(t)


def _section_executive_summary(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("1. Executive Summary", styles["section_h1"]))

    model   = session_data.get("model")
    acc     = session_data.get("accuracy")
    areas   = session_data.get("class_areas")
    cfg     = session_data.get("config") or {}

    gate_passed = _g(model, "gate_passed")
    gate_color  = _GREEN if gate_passed == "Yes" else _RED

    rows = [
        ("Model type",          _g(model, "model_type")),
        ("Overall Accuracy",    _g(model, "oa")),
        ("Cohen's Kappa",       _g(model, "kappa")),
        ("Minority F1",         _g(model, "minority_f1")),
        ("SMOTE applied",       _g(model, "smote_applied")),
        ("Quality gate",        "PASSED ✓" if gate_passed == "Yes"
                                else "FAILED ✗" if gate_passed != "N/A" else "N/A"),
        ("Total area (ha)",     _g(areas, "total_area_ha")),
        ("Independent OA",      _g(acc, "oa")),
        ("Independent Kappa",   _g(acc, "kappa")),
    ]

    story.append(_two_col_table(rows))

    gate_msg = _g(model, "gate_message")
    if gate_msg != "N/A":
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"Gate message: {gate_msg}", styles["caption"]))


def _section_classification(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("2. Classification Results", styles["section_h1"]))

    model = session_data.get("model")

    # ── 2a. Model parameters ──────────────────────────────────────────────
    story.append(Paragraph("2a. Model Parameters", styles["section_h2"]))
    params = [
        ("Model type",      _g(model, "model_type")),
        ("N° estimators",   _g(model, "n_estimators")),
        ("Max depth",       _g(model, "max_depth")),
        ("CV folds",        _g(model, "k_folds")),
        ("SMOTE applied",   _g(model, "smote_applied")),
        ("Random state",    _g(model, "random_state")),
    ]
    story.append(_two_col_table(params))

    # ── 2b. Global metrics ────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("2b. Global Metrics", styles["section_h2"]))
    metrics = [
        ("Overall Accuracy (OA)", _g(model, "oa")),
        ("Cohen's Kappa",         _g(model, "kappa")),
        ("Minority F1",           _g(model, "minority_f1")),
    ]
    story.append(_two_col_table(metrics))

    # ── 2c. Per-class metrics ─────────────────────────────────────────────
    per_class = None
    if model is not None:
        per_class = getattr(model, "per_class_metrics", None)

    if per_class:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("2c. Per-class Metrics", styles["section_h2"]))
        header = ["Class", "Precision", "Recall", "F1", "Support"]
        rows   = [header]
        for cls_id in sorted(per_class.keys()):
            m = per_class[cls_id]
            rows.append([
                str(cls_id),
                f"{m.get('precision', 0):.4f}",
                f"{m.get('recall', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                str(int(m.get("support", 0))),
            ])
        col_w = [3 * cm, 3 * cm, 3 * cm, 3 * cm, 3 * cm]
        t = Table(rows, colWidths=col_w)
        t.setStyle(_HEADER_STYLE)
        story.append(t)

    # ── 2d. CV scores ─────────────────────────────────────────────────────
    cv_scores = None
    if model is not None:
        cv_scores = getattr(model, "cv_scores", None)

    if cv_scores:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("2d. Cross-validation Scores", styles["section_h2"]))
        header = ["Fold"] + [str(i + 1) for i in range(len(cv_scores))]
        data   = [header, ["OA"] + [f"{s:.4f}" for s in cv_scores]]
        t      = Table(data)
        t.setStyle(_HEADER_STYLE)
        story.append(t)
        mean_oa = sum(cv_scores) / len(cv_scores)
        story.append(Paragraph(
            f"Mean: {mean_oa:.4f}  |  "
            f"Min: {min(cv_scores):.4f}  |  "
            f"Max: {max(cv_scores):.4f}",
            styles["caption"],
        ))


def _section_confidence_stats(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("3. Confidence Statistics", styles["section_h1"]))

    cs = session_data.get("confidence_stats")

    if cs is None:
        story.append(Paragraph(
            "Confidence raster statistics not available for this run. "
            "Re-run ML Classification to generate a confidence map.",
            styles["body"],
        ))
        return

    threshold = cs.get("threshold", "N/A")
    rows = [
        ("P05 (5th percentile)",         f"{cs['p05']:.4f}" if "p05" in cs else "N/A"),
        ("P50 (median)",                  f"{cs['p50']:.4f}" if "p50" in cs else "N/A"),
        ("P95 (95th percentile)",         f"{cs['p95']:.4f}" if "p95" in cs else "N/A"),
        ("Mean confidence",               f"{cs['mean']:.4f}" if "mean" in cs else "N/A"),
        ("Min / Max",
         f"{cs.get('min', float('nan')):.4f} / {cs.get('max', float('nan')):.4f}"),
        ("Confidence threshold",          str(threshold)),
        ("Pixels above threshold",        f"{cs.get('n_above', 'N/A'):,}"
                                          if isinstance(cs.get("n_above"), int) else "N/A"),
        ("% above threshold",             f"{cs['pct_above']:.1f}%"
                                          if "pct_above" in cs else "N/A"),
        ("Valid pixels",                  f"{cs.get('n_valid', 'N/A'):,}"
                                          if isinstance(cs.get("n_valid"), int) else "N/A"),
    ]
    story.append(_two_col_table(rows))
    story.append(Paragraph(
        "Percentiles estimated via reservoir sampling (100,000 pixel subsample).",
        styles["caption"],
    ))


def _section_postprocess_params(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("4. Post-Processing Parameters", styles["section_h1"]))

    cfg   = session_data.get("config") or {}
    chain = session_data.get("postprocess_chain")

    # ── 4a. Chain steps ───────────────────────────────────────────────────
    story.append(Paragraph("4a. Chain Steps Applied", styles["section_h2"]))

    _STEP_LABELS = {
        "confidence_filter":   "Confidence filter",
        "median_smooth":       "Median smooth",
        "morphological_close": "Morphological closing",
        "sieve_filter":        "Sieve filter (MMU)",
    }

    if chain:
        step_rows = [["Step", "Output file", "Applied"]]
        for step_key, label in _STEP_LABELS.items():
            out = chain.get(step_key)
            applied = "Yes" if out else "No"
            fname   = Path(str(out)).name if out else "—"
            step_rows.append([label, fname, applied])
        t = Table(step_rows, colWidths=[5 * cm, 8 * cm, 3 * cm])
        t.setStyle(_HEADER_STYLE)
        story.append(t)

        final = chain.get("final")
        if final:
            story.append(Paragraph(
                f"Final output: {Path(str(final)).name}",
                styles["caption"],
            ))
    else:
        story.append(Paragraph(
            "Post-processing chain was not executed for this run.",
            styles["body"],
        ))

    # ── 4b. Parameters from config ────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("4b. Parameters (from pipeline_config.yaml)", styles["section_h2"]))
    param_rows = [
        ("Confidence threshold",         str(cfg.get("confidence_threshold",    "N/A"))),
        ("Min mapping unit (ha)",        str(cfg.get("min_mapping_unit_ha",     "N/A"))),
        ("Sieve connectivity",           str(cfg.get("sieve_connectivity",      "N/A"))),
        ("Morphological kernel (px)",    str(cfg.get("morpho_kernel_size",      "N/A"))),
        ("Drift alert threshold (%)",    str(cfg.get("drift_alert_pct",         "N/A"))),
    ]
    story.append(_two_col_table(param_rows))


def _section_quality_gate(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("5. Quality Gate Results", styles["section_h1"]))

    model = session_data.get("model")
    acc   = session_data.get("accuracy")
    cfg   = session_data.get("config") or {}

    min_oa  = float(cfg.get("min_oa_threshold", 0.80))
    min_f1  = float(cfg.get("min_minority_f1",  0.70))
    min_k   = 0.60

    def _gate_row(label: str, value_str: str, threshold: float,
                  higher_is_better: bool = True) -> list[str]:
        try:
            val = float(value_str)
            if higher_is_better:
                status = "PASS" if val >= threshold else "FAIL"
            else:
                status = "PASS" if val <= threshold else "FAIL"
        except (ValueError, TypeError):
            status = "N/A"
        return [label, value_str, f"{threshold:.4f}", _gate_badge(status)]

    header = ["Metric", "Value", "Threshold", "Result"]
    rows   = [header]

    # Training gate
    rows.append(["— Training gate —", "", "", ""])
    rows.append(_gate_row(
        "Overall Accuracy (training)", _g(model, "oa"), min_oa
    ))
    rows.append(_gate_row(
        "Minority F1 (training)", _g(model, "minority_f1"), min_f1
    ))
    rows.append(_gate_row(
        "Cohen's Kappa (training)", _g(model, "kappa"), min_k
    ))

    # Independent accuracy gate (if available)
    if acc is not None:
        rows.append(["— Independent accuracy —", "", "", ""])
        rows.append(_gate_row(
            "Overall Accuracy (independent)", _g(acc, "oa"), min_oa
        ))
        rows.append(_gate_row(
            "Cohen's Kappa (independent)", _g(acc, "kappa"), min_k
        ))

    col_w = [6 * cm, 3 * cm, 3 * cm, 4 * cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_HEADER_STYLE)
    story.append(t)

    # Overall gate verdict
    gate_passed = _g(model, "gate_passed")
    if gate_passed == "Yes":
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Overall quality gate: PASSED ✓", styles["body"]))
    elif gate_passed == "No":
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Overall quality gate: FAILED ✗", styles["body"]))


def _section_class_areas(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("6. Class Area Distribution", styles["section_h1"]))

    areas = session_data.get("class_areas")

    if areas is None:
        story.append(Paragraph(
            "Class area statistics were not computed for this run. "
            "Run 'Compute Class Areas' in Post-processing to generate them.",
            styles["body"],
        ))
        return

    areas_ha   = getattr(areas, "areas_ha",   {}) or {}
    pix_counts = getattr(areas, "pixel_counts", {}) or {}
    total_ha   = getattr(areas, "total_area_ha", None)
    px_res     = getattr(areas, "pixel_res_m", None)

    if not areas_ha:
        story.append(Paragraph("No class area data available.", styles["body"]))
        return

    header = ["Class", "Pixels", "Area (ha)", "Area (%)"]
    rows   = [header]
    for cls_id in sorted(areas_ha.keys()):
        ha  = areas_ha[cls_id]
        pct = (ha / total_ha * 100.0) if (total_ha and total_ha > 0) else 0.0
        rows.append([
            str(cls_id),
            f"{pix_counts.get(cls_id, 0):,}",
            f"{ha:.2f}",
            f"{pct:.1f}%",
        ])

    col_w = [3 * cm, 4 * cm, 4 * cm, 4 * cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(_HEADER_STYLE)
    story.append(t)

    res_str = f"{px_res:.2f} m" if px_res is not None else "N/A"
    story.append(Paragraph(
        f"Total mapped area: {total_ha:.2f} ha  |  "
        f"{len(areas_ha)} class(es)  |  "
        f"Pixel resolution: {res_str}"
        if total_ha is not None
        else f"Pixel resolution: {res_str}",
        styles["caption"],
    ))


def _section_preprocessing(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("7. Pipeline Configuration", styles["section_h1"]))

    cfg = session_data.get("config") or {}

    rows = [
        ("Target CRS",                  str(cfg.get("target_crs",           "N/A"))),
        ("Resolution (m/px)",           str(cfg.get("target_resolution_m",  "N/A"))),
        ("ENL (default SAR)",           str(cfg.get("sar_enl_default",      "N/A"))),
        ("ENL Sentinel-1 IW",           str(cfg.get("sar_enl_sentinel1_iw", "N/A"))),
        ("Min overlap (%)",             str(cfg.get("min_overlap_pct",      "N/A"))),
        ("Max resolution ratio (×)",    str(cfg.get("max_resolution_ratio", "N/A"))),
        ("Max date gap (days)",         str(cfg.get("max_date_gap_days",    "N/A"))),
        ("Min OA threshold (gate)",     str(cfg.get("min_oa_threshold",     "N/A"))),
        ("Min F1 threshold (gate)",     str(cfg.get("min_minority_f1",      "N/A"))),
        ("Coreg RMSE threshold (px)",   str(cfg.get("coreg_rmse_threshold", "N/A"))),
    ]
    story.append(_two_col_table(rows))

    # Independent accuracy assessment
    acc = session_data.get("accuracy")
    if acc is not None:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph(
            "Accuracy Assessment (reference points)", styles["section_h2"]
        ))
        acc_rows = [
            ("Overall Accuracy",    _g(acc, "oa")),
            ("Cohen's Kappa",       _g(acc, "kappa")),
            ("Total points",        _g(acc, "n_points")),
            ("Valid points",        _g(acc, "n_valid")),
            ("Discarded points",    _g(acc, "n_discarded")),
        ]
        story.append(_two_col_table(acc_rows))


def _section_audit_log(
    styles: dict,
    run_id: str,
    story: list,
) -> None:
    story.append(Paragraph("8. Audit Log", styles["section_h1"]))

    # Import here to keep pipeline boundary clean (no circular imports)
    try:
        from pipeline import audit as _audit
        events: list[dict] = _audit.get_log(run_id)
    except Exception:
        events = []

    if not events:
        story.append(Paragraph(
            "No audit events found for this run.", styles["body"]
        ))
        return

    truncated = len(events) > MAX_AUDIT_ROWS
    display   = events[-MAX_AUDIT_ROWS:] if truncated else events

    header = ["#", "Type", "Stage / Action", "Decision", "Timestamp"]
    rows   = [header]
    for i, ev in enumerate(display, start=1):
        payload   = ev.get("payload") or {}
        stage_val = payload.get("stage") or payload.get("action") or "—"
        # Truncate long stage strings
        stage_val = textwrap.shorten(str(stage_val), width=30, placeholder="…")
        rows.append([
            str(i),
            str(ev.get("event_type", "—")),
            stage_val,
            str(ev.get("decision", "—")),
            str(ev.get("timestamp", "—"))[:19],
        ])

    col_w = [1.2 * cm, 2.5 * cm, 5 * cm, 2.5 * cm, 4.5 * cm]
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(_HEADER_STYLE)
    story.append(t)

    if truncated:
        story.append(Paragraph(
            f"(Showing the last {MAX_AUDIT_ROWS} of {len(events)} events. "
            "See the full audit_log.json for the complete history.)",
            styles["caption"],
        ))


# ── Public API ────────────────────────────────────────────────────────────────

def generate_report(
    run_id:        str,
    session_data:  dict,
    out_path:      Path,
    operator_name: str = "",
) -> Path:
    """
    Generate a multi-section technical PDF report for a TerraForge run.

    Parameters
    ----------
    run_id        : Run identifier (e.g. "A1B2C3D4").
    session_data  : Shallow snapshot of pipeline session state.  All keys are
                    optional; missing values render as "N/A" (Not Available).
    out_path      : Destination path for the PDF (parent must exist or will be
                    created).
    operator_name : Name of the operator signing the report.

    Returns
    -------
    Path  — resolved path to the written PDF file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    styles       = _build_styles()
    footer_cb    = _make_footer_cb(run_id, generated_at)

    # ── Document template ─────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.5 * cm,   # leave room for footer
    )

    frame = Frame(
        MARGIN, 2.5 * cm,
        PAGE_W - 2 * MARGIN,
        PAGE_H - MARGIN - 2.5 * cm,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=footer_cb)
    doc.addPageTemplates([template])

    # ── Story ─────────────────────────────────────────────────────────────────
    from reportlab.platypus import PageBreak

    story: list = []

    _section_cover(styles, run_id, operator_name, generated_at, story)
    story.append(PageBreak())

    _section_executive_summary(styles, session_data, story)
    story.append(PageBreak())

    _section_classification(styles, session_data, story)
    story.append(PageBreak())

    _section_confidence_stats(styles, session_data, story)
    story.append(PageBreak())

    _section_postprocess_params(styles, session_data, story)
    story.append(PageBreak())

    _section_quality_gate(styles, session_data, story)
    story.append(PageBreak())

    _section_class_areas(styles, session_data, story)
    story.append(PageBreak())

    _section_preprocessing(styles, session_data, story)
    story.append(PageBreak())

    _section_audit_log(styles, run_id, story)

    doc.build(story)
    return out_path.resolve()
