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
# Mining-specific additions
# ============================================================================

# Mining class colours (hex) — used for classification map legend
_MINING_COLOURS: dict[int, str] = {
    # Pit wall
    1: "#8B4513",  # Exposed Fresh Rock — brown
    2: "#DAA520",  # Weathered/Oxidised — goldenrod
    3: "#DC143C",  # Instability Zone — crimson (risk)
    4: "#228B22",  # Vegetation — forest green
    5: "#4169E1",  # Water/Seepage — royal blue
    # TSF extras
    6: "#A0522D",  # Seepage/Discolouration
}

_GOLD = colors.HexColor("#C8860A")


def _raster_preview_image(
    raster_path,
    tmp_png: Path,
    discrete: bool = True,
    class_colours: dict | None = None,
) -> Path | None:
    """
    Render a small preview PNG from a GeoTIFF using matplotlib.
    Returns the PNG path or None if rendering fails.
    """
    try:
        import numpy as np
        import rasterio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        with rasterio.open(raster_path) as src:
            data = src.read(1)
            nodata = src.nodata

        if nodata is not None:
            data = data.astype(float)
            data[data == nodata] = np.nan

        fig, ax = plt.subplots(figsize=(6, 4))

        if discrete and class_colours:
            valid_classes = sorted(
                c for c in class_colours if np.any(data == c)
            )
            bounds = [c - 0.5 for c in valid_classes] + [valid_classes[-1] + 0.5]
            hex_list = [class_colours[c] for c in valid_classes]
            cmap = mcolors.ListedColormap(hex_list)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            ax.imshow(data, cmap=cmap, norm=norm, interpolation="nearest")
        else:
            finite = data[np.isfinite(data)]
            vmin = float(np.nanpercentile(finite, 2))
            vmax = float(np.nanpercentile(finite, 98))
            ax.imshow(data, cmap="RdYlGn", vmin=vmin, vmax=vmax, interpolation="bilinear")

        ax.axis("off")
        ax.set_facecolor("#1B2A4A")
        fig.patch.set_facecolor("#1B2A4A")
        plt.tight_layout(pad=0)
        fig.savefig(tmp_png, dpi=120, bbox_inches="tight", facecolor="#1B2A4A")
        plt.close(fig)
        return tmp_png
    except Exception:
        return None


def _section_mining_cover(
    styles: dict,
    run_id: str,
    site_name: str,
    operator: str,
    survey_type: str,
    mode: str,
    generated_at: str,
    story: list,
) -> None:
    from reportlab.platypus import HRFlowable

    # Coloured header bar
    story.append(Spacer(1, 0.5 * cm))

    # Logo placeholder with mining colour
    logo = _LogoPlaceholder()
    story.append(logo)
    story.append(Spacer(1, 0.8 * cm))

    # Title
    mode_title = "Open Pit Wall Assessment" if mode == "pit_wall" else "Tailings Storage Facility Survey"
    story.append(Paragraph("Geospatial Material Classification Report", styles["cover_title"]))
    story.append(Paragraph(mode_title, styles["cover_sub"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width="100%", thickness=2, color=_GOLD))
    story.append(Spacer(1, 0.6 * cm))

    cover_data = [
        ("Site Name",    site_name),
        ("Operator",     operator or "N/A"),
        ("Date",         generated_at[:10]),
        ("Run ID",       run_id),
        ("Survey Type",  survey_type),
        ("Mode",         mode.replace("_", " ").title()),
    ]
    t = Table(
        [[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
         for k, v in cover_data],
        colWidths=[5 * cm, 11 * cm],
    )
    t.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), ["#F8F9FA", "#FFFFFF"]),
    ]))
    story.append(t)
    story.append(Spacer(1, 1.0 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=_NAVY))
    story.append(Spacer(1, 0.4 * cm))

    disclaimer = (
        "This report is produced by an automated geospatial pipeline. "
        "Results should be reviewed by a qualified geotechnical engineer "
        "before operational decisions are made."
    )
    story.append(Paragraph(f"<b>Disclaimer:</b> {disclaimer}", styles["caption"]))


def _section_mining_executive_summary(
    styles: dict,
    clf_result,
    area_result,
    class_defs: dict,
    cfg: dict,
    story: list,
) -> None:
    from reportlab.platypus import HRFlowable

    story.append(Paragraph("1. Executive Summary", styles["section_h1"]))

    oa      = getattr(clf_result, "oa", None)
    kappa   = getattr(clf_result, "kappa", None)
    mf1     = getattr(clf_result, "minority_f1", None)
    passed  = getattr(clf_result, "gate_passed", None)
    msg     = getattr(clf_result, "gate_message", "")

    # Gate badge row
    gate_label = "PASS" if passed else "FAIL"
    gate_col   = _GREEN if passed else _RED

    kv_rows = [
        ("Overall Accuracy (OA)",  f"{oa:.4f}" if oa is not None else "N/A"),
        ("Cohen's Kappa",           f"{kappa:.4f}" if kappa is not None else "N/A"),
        ("Minority Class F1",       f"{mf1:.4f}" if mf1 is not None else "N/A"),
        ("Quality Gate",            gate_label),
        ("Model type",              getattr(clf_result, "model_type", "N/A")),
        ("SMOTE applied",           str(getattr(clf_result, "smote_applied", "N/A"))),
    ]
    t = Table(
        [[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])]
         for k, v in kv_rows],
        colWidths=[7 * cm, 9 * cm],
    )
    ts = TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), ["#F8F9FA", "#FFFFFF"]),
    ])
    # Colour gate row
    gate_row_idx = 3
    ts.add("TEXTCOLOR", (1, gate_row_idx), (1, gate_row_idx), gate_col)
    ts.add("FONTNAME",  (1, gate_row_idx), (1, gate_row_idx), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))
    if msg:
        story.append(Paragraph(f"Gate: {msg}", styles["caption"]))
    story.append(Spacer(1, 0.3 * cm))

    # Class area summary table
    if area_result is not None:
        areas_ha  = getattr(area_result, "areas_ha",   {}) or {}
        areas_pct = getattr(area_result, "areas_pct",  {}) or {}
        total_ha  = getattr(area_result, "total_area_ha", None)

        header = ["Class", "Name", "Area (ha)", "Area (%)"]
        rows   = [header]
        for cls_id in sorted(areas_ha.keys()):
            ha  = areas_ha.get(cls_id, 0.0)
            pct = areas_pct.get(cls_id, 0.0)
            name = class_defs.get(str(cls_id), class_defs.get(cls_id, f"Class {cls_id}"))
            rows.append([str(cls_id), str(name), f"{ha:.2f}", f"{pct:.1f}%"])
        if total_ha:
            rows.append(["", "TOTAL", f"{total_ha:.2f}", "100.0%"])

        col_w = [2*cm, 8*cm, 3*cm, 3*cm]
        tbl = Table(rows, colWidths=col_w)
        tbl.setStyle(_HEADER_STYLE)
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # Auto-generated executive paragraph
    if oa is not None and area_result is not None:
        areas_ha  = getattr(area_result, "areas_ha",   {}) or {}
        areas_pct = getattr(area_result, "areas_pct",  {}) or {}
        total_ha  = getattr(area_result, "total_area_ha", None) or 0.0
        drift_pct = cfg.get("drift_alert_pct", 20)
        n_classes = len(areas_ha)
        dominant_cls = max(areas_pct, key=areas_pct.get) if areas_pct else "N/A"
        dominant_pct = areas_pct.get(dominant_cls, 0.0)
        dominant_name = class_defs.get(str(dominant_cls), f"Class {dominant_cls}")
        para = (
            f"Classification achieved an Overall Accuracy of <b>{oa:.1%}</b> "
            f"(Kappa: <b>{kappa:.3f}</b>) across <b>{n_classes}</b> material "
            f"classes covering <b>{total_ha:.1f} ha</b> of surveyed area. "
            f"The dominant class is <i>{dominant_name}</i> at "
            f"<b>{dominant_pct:.1f}%</b> of the surveyed area. "
            f"Drift monitoring threshold is set at {drift_pct}% per class."
        )
        story.append(Paragraph(para, styles["body"]))


def _section_mining_classification_map(
    styles: dict,
    cog_path,
    class_defs: dict,
    area_result,
    story: list,
) -> None:
    import tempfile, os
    story.append(Paragraph("2. Material Classification Map", styles["section_h1"]))

    # Preview image
    if cog_path and Path(cog_path).exists():
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_png = Path(f.name)

        preview = _raster_preview_image(
            cog_path, tmp_png, discrete=True,
            class_colours=_MINING_COLOURS,
        )
        if preview and preview.exists():
            from reportlab.platypus import Image as RLImage
            avail_w = PAGE_W - 2 * MARGIN
            story.append(RLImage(str(preview), width=avail_w, height=avail_w * 0.6))
            story.append(Spacer(1, 0.2 * cm))
            try:
                os.unlink(tmp_png)
            except OSError:
                pass
        else:
            story.append(Paragraph(
                "Classification map preview unavailable (matplotlib not installed).",
                styles["caption"],
            ))
    else:
        story.append(Paragraph("Classified raster not available.", styles["caption"]))

    # Legend table
    areas_ha  = getattr(area_result, "areas_ha",   {}) or {} if area_result else {}
    areas_pct = getattr(area_result, "areas_pct",  {}) or {} if area_result else {}

    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import Table as RLTable

    header = ["Colour", "Class ID", "Class Name", "Area (ha)", "Area (%)"]
    rows   = [header]
    for cls_id in sorted(class_defs.keys()):
        hex_c = _MINING_COLOURS.get(int(cls_id) if str(cls_id).isdigit() else 0, "#AAAAAA")
        name  = class_defs[cls_id]
        ha    = areas_ha.get(int(cls_id), areas_ha.get(str(cls_id), 0.0))
        pct   = areas_pct.get(int(cls_id), areas_pct.get(str(cls_id), 0.0))
        rows.append(["", str(cls_id), str(name), f"{ha:.2f}", f"{pct:.1f}%"])

    col_w = [1.5*cm, 2*cm, 7*cm, 3*cm, 2.5*cm]
    tbl   = Table(rows, colWidths=col_w)
    base_style = list(_HEADER_STYLE.getCommands())
    # Colour swatches in first column (body rows)
    for i, cls_id in enumerate(sorted(class_defs.keys()), start=1):
        hex_c = _MINING_COLOURS.get(int(cls_id) if str(cls_id).isdigit() else 0, "#AAAAAA")
        base_style.append(("BACKGROUND", (0, i), (0, i), rl_colors.HexColor(hex_c)))
    tbl.setStyle(TableStyle(base_style))
    story.append(tbl)


def _section_mining_confidence(
    styles: dict,
    confidence_path,
    story: list,
) -> None:
    import tempfile, os
    story.append(Paragraph("3. Confidence Analysis", styles["section_h1"]))

    if confidence_path is None or not Path(confidence_path).exists():
        story.append(Paragraph("Confidence raster not available for this run.", styles["body"]))
        return

    # Compute percentile stats
    stats = compute_confidence_stats(Path(confidence_path), threshold=0.60)

    if stats:
        kv = [
            ("Mean confidence",       f"{stats.get('mean', 0):.3f}"),
            ("Median (p50)",          f"{stats.get('p50', 0):.3f}"),
            ("p05 / p95",             f"{stats.get('p05', 0):.3f} / {stats.get('p95', 0):.3f}"),
            ("Low confidence < 0.60", f"{stats.get('low_conf_pct', 0):.1f}% of pixels"),
        ]
        story.append(_two_col_table(kv))
        story.append(Spacer(1, 0.3 * cm))
        low_pct = stats.get("low_conf_pct", 0)
        if low_pct > 15:
            story.append(Paragraph(
                f"<b>Note:</b> {low_pct:.1f}% of pixels have confidence < 0.60. "
                "These areas require field verification.",
                styles["caption"],
            ))

    # Confidence map preview
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_png = Path(f.name)
    preview = _raster_preview_image(confidence_path, tmp_png, discrete=False)
    if preview and preview.exists():
        from reportlab.platypus import Image as RLImage
        avail_w = PAGE_W - 2 * MARGIN
        story.append(RLImage(str(preview), width=avail_w, height=avail_w * 0.5))
        story.append(Paragraph(
            "Confidence map (RdYlGn): green = high confidence, red = low confidence.",
            styles["caption"],
        ))
        try:
            os.unlink(tmp_png)
        except OSError:
            pass


def _section_mining_methodology(
    styles: dict,
    clf_result,
    feat_names: list,
    class_defs: dict,
    cfg: dict,
    story: list,
) -> None:
    story.append(Paragraph("5. Methodology", styles["section_h1"]))

    # Model info
    model_type = getattr(clf_result, "model_type", "N/A")
    k_folds    = cfg.get("default_k_folds", 5)
    smote      = getattr(clf_result, "smote_applied", False)
    cv_scores  = getattr(clf_result, "cv_scores", [])

    kv = [
        ("Classifier",           model_type),
        ("Cross-validation",     f"{k_folds}-fold stratified"),
        ("SMOTE resampling",     str(smote)),
        ("CV fold OA scores",    ", ".join(f"{s:.3f}" for s in cv_scores) or "N/A"),
        ("Min mapping unit",     f"{cfg.get('min_mapping_unit_ha', 0.5)} ha"),
        ("Coreg RMSE threshold", f"{cfg.get('coreg_rmse_threshold', 0.5)} px"),
        ("OA threshold",         f"{cfg.get('min_oa_threshold', 0.80):.0%}"),
        ("Minority F1 threshold",f"{cfg.get('min_minority_f1', 0.70):.0%}"),
    ]
    story.append(_two_col_table(kv))
    story.append(Spacer(1, 0.3 * cm))

    # Feature list
    if feat_names:
        story.append(Paragraph("<b>Features computed:</b>", styles["body"]))
        feat_str = ", ".join(feat_names)
        story.append(Paragraph(feat_str, styles["caption"]))
        story.append(Spacer(1, 0.2 * cm))

    # Feature importances
    fi = getattr(clf_result, "feature_importances", None)
    if fi:
        story.append(Paragraph("<b>Top feature importances:</b>", styles["body"]))
        top = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        rows = [["Feature", "Importance"]]
        for name, imp in top:
            rows.append([name, f"{imp:.4f}"])
        col_w = [9 * cm, 7 * cm]
        tbl = Table(rows, colWidths=col_w)
        tbl.setStyle(_HEADER_STYLE)
        story.append(tbl)


def _section_mining_audit(
    styles: dict,
    run_id: str,
    audit_path,
    story: list,
) -> None:
    story.append(Paragraph("6. Audit Trail", styles["section_h1"]))

    # Load audit log from JSON if provided, else fall back to get_log
    log: list[dict] = []
    if audit_path and Path(audit_path).exists():
        import json
        try:
            log = json.loads(Path(audit_path).read_text(encoding="utf-8"))
        except Exception:
            pass

    if not log:
        from pipeline.audit import get_log
        log = get_log(run_id)

    if not log:
        story.append(Paragraph("No audit events recorded for this run.", styles["body"]))
        return

    header = ["Timestamp", "Stage", "Decision"]
    rows   = [header]
    for entry in log[:MAX_AUDIT_ROWS]:
        ts       = entry.get("timestamp", "")[:19].replace("T", " ")
        stage    = entry.get("event_type", "")
        decision = entry.get("decision") or str(entry.get("details", {}).get("stage", ""))
        rows.append([ts, stage, decision or "—"])

    col_w = [5 * cm, 4 * cm, 7 * cm]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(_HEADER_STYLE)
    story.append(tbl)

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"This pipeline run is fully traceable. "
        f"The complete audit log is available in <b>{run_id}_audit_log.json</b>. "
        f"Reproduce this analysis using TerraForge Mining Intelligence "
        f"with run ID <b>{run_id}</b>.",
        styles["caption"],
    ))


def generate_report(
    run_id: str,
    out_path,
    session_data: dict | None = None,
    operator_name: str = "",
    # Mining-specific kwargs (used when session_data is None)
    site_name: str = "",
    operator: str = "",
    survey_type: str = "Satellite",
    mode: str = "pit_wall",
    cog_path=None,
    confidence_path=None,
    classification_result=None,
    area_result=None,
    class_defs: dict | None = None,
    cfg: dict | None = None,
    audit_path=None,
) -> Path:
    """
    Generate a TerraForge mining PDF report.

    Supports two call styles:

    Legacy (Streamlit app):
        generate_report(run_id, session_data, out_path, operator_name="")

    Mining demo:
        generate_report(run_id, out_path, site_name=..., operator=...,
                        survey_type=..., mode=..., cog_path=..., ...)

    All parameters are optional; missing values render as 'N/A'.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    styles       = _build_styles()
    _cfg         = cfg or {}

    # ── Determine which rendering path to use ─────────────────────────────────
    _mining_mode = (session_data is None)

    _site_name   = site_name or (session_data or {}).get("site_name", "")
    _operator    = operator or operator_name or ""
    _survey_type = survey_type
    _mode        = mode
    _cog_path    = cog_path
    _conf_path   = confidence_path
    _clf         = classification_result
    _area        = area_result
    _class_defs  = class_defs or {}

    footer_label = _site_name or "TerraForge Mining Intelligence"
    footer_cb    = _make_footer_cb(run_id, generated_at)

    # ── Document template ─────────────────────────────────────────────────────
    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=2.5 * cm,
    )
    frame = Frame(
        MARGIN, 2.5 * cm,
        PAGE_W - 2 * MARGIN,
        PAGE_H - MARGIN - 2.5 * cm,
        id="main",
    )
    template = PageTemplate(id="main", frames=[frame], onPage=footer_cb)
    doc.addPageTemplates([template])

    from reportlab.platypus import PageBreak
    story: list = []

    if _mining_mode:
        # ── Mining demo report ────────────────────────────────────────────────
        _section_mining_cover(
            styles, run_id, _site_name, _operator, _survey_type, _mode,
            generated_at, story,
        )
        story.append(PageBreak())

        _section_mining_executive_summary(
            styles, _clf, _area, _class_defs, _cfg, story,
        )
        story.append(PageBreak())

        _section_mining_classification_map(
            styles, _cog_path, _class_defs, _area, story,
        )
        story.append(PageBreak())

        _section_mining_confidence(styles, _conf_path, story)
        story.append(PageBreak())

        # Change detection placeholder (no reference available in standalone demo)
        story.append(Paragraph("4. Change Detection", styles["section_h1"]))
        story.append(Paragraph(
            "No reference survey is available for this run. "
            "Change detection requires a prior classified raster to compute "
            "per-class area deltas. "
            "Re-run with --reference-dir to enable drift monitoring.",
            styles["body"],
        ))
        story.append(PageBreak())

        _section_mining_methodology(
            styles, _clf,
            getattr(_clf, "feature_names", []) if _clf else [],
            _class_defs, _cfg, story,
        )
        story.append(PageBreak())

        _section_mining_audit(styles, run_id, audit_path, story)

    else:
        # ── Legacy session_data report (Streamlit app) ────────────────────────
        _section_cover(styles, run_id, _operator, generated_at, story)
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
