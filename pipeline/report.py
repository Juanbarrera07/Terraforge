"""
pipeline/report.py — Technical PDF report generator for TerraForge.

Produces a multi-section corporate PDF from pipeline session data using
reportlab.  All work is disk-based: no pixel arrays are held in memory.

Public API
----------
generate_report(run_id, session_data, out_path, operator_name="") -> Path

Session data keys consumed (all optional — missing keys yield "N/D")
---------------------------------------------------------------------
"model"    : ClassificationResult  — OA, Kappa, F1, model type, SMOTE, CV
"accuracy" : AccuracyResult        — independent accuracy assessment
"areas"    : ClassAreaResult       — per-class hectare areas
"config"   : dict                  — pipeline_config.yaml values

Design rules
------------
- No Streamlit imports.
- Every dict/attr access goes through _g() — the report never raises on
  missing or None session data.
- Audit log rows are capped at MAX_AUDIT_ROWS to keep file size predictable.
"""
from __future__ import annotations

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
_LIGHT = colors.HexColor("#F1F5F9")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_AUDIT_ROWS = 50
PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm


# ── Safe accessor ─────────────────────────────────────────────────────────────

def _g(obj: Any, *keys, default: str = "N/D") -> str:
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
        return "Sí" if cur else "No"
    return str(cur)


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
    data = [["Parámetro", "Valor"]] + list(pairs)
    t = Table(data, colWidths=col_widths)
    t.setStyle(_HEADER_STYLE)
    return t


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
    story.append(Paragraph("Reporte Técnico de Clasificación", styles["cover_title"]))
    story.append(Paragraph("TerraForge Mining Intelligence", styles["cover_sub"]))
    story.append(Spacer(1, 0.8 * cm))

    cover_data = [
        ("Run ID",        run_id),
        ("Operador",      operator_name or "—"),
        ("Fecha/Hora",    generated_at),
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
    story.append(Paragraph("1. Resumen Ejecutivo", styles["section_h1"]))

    model   = session_data.get("model")
    acc     = session_data.get("accuracy")
    areas   = session_data.get("areas")
    cfg     = session_data.get("config") or {}

    gate_passed = _g(model, "gate_passed")
    gate_color  = _GREEN if gate_passed == "Sí" else _RED

    rows = [
        ("Tipo de modelo",      _g(model, "model_type")),
        ("Overall Accuracy",    _g(model, "oa")),
        ("Cohen's Kappa",       _g(model, "kappa")),
        ("Minority F1",         _g(model, "minority_f1")),
        ("SMOTE aplicado",      _g(model, "smote_applied")),
        ("Quality gate",        "PASSED ✓" if gate_passed == "Sí" else "FAILED ✗"
                                if gate_passed != "N/D" else "N/D"),
        ("Área total (ha)",     _g(areas, "total_area_ha")),
        ("OA independiente",    _g(acc, "oa")),
        ("Kappa independiente", _g(acc, "kappa")),
    ]

    story.append(_two_col_table(rows))

    gate_msg = _g(model, "gate_message")
    if gate_msg != "N/D":
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph(f"Mensaje del gate: {gate_msg}", styles["caption"]))


def _section_classification(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("2. Resultados de Clasificación", styles["section_h1"]))

    model = session_data.get("model")

    # ── 2a. Model parameters ──────────────────────────────────────────────
    story.append(Paragraph("2a. Parámetros del modelo", styles["section_h2"]))
    params = [
        ("Tipo de modelo",    _g(model, "model_type")),
        ("N° estimadores",    _g(model, "n_estimators")),
        ("Max depth",         _g(model, "max_depth")),
        ("CV folds",          _g(model, "k_folds")),
        ("SMOTE aplicado",    _g(model, "smote_applied")),
        ("Random state",      _g(model, "random_state")),
    ]
    story.append(_two_col_table(params))

    # ── 2b. Global metrics ────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("2b. Métricas globales", styles["section_h2"]))
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
        story.append(Paragraph("2c. Métricas por clase", styles["section_h2"]))
        header = ["Clase", "Precision", "Recall", "F1", "Support"]
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
        story.append(Paragraph("2d. Scores de validación cruzada", styles["section_h2"]))
        header = ["Fold"] + [str(i + 1) for i in range(len(cv_scores))]
        data   = [header, ["OA"] + [f"{s:.4f}" for s in cv_scores]]
        t      = Table(data)
        t.setStyle(_HEADER_STYLE)
        story.append(t)
        mean_oa = sum(cv_scores) / len(cv_scores)
        story.append(Paragraph(
            f"Media: {mean_oa:.4f}  |  "
            f"Min: {min(cv_scores):.4f}  |  "
            f"Max: {max(cv_scores):.4f}",
            styles["caption"],
        ))

    # ── 2e. Class areas ───────────────────────────────────────────────────
    areas = session_data.get("areas")
    if areas is not None:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("2e. Áreas por clase", styles["section_h2"]))
        areas_ha    = getattr(areas, "areas_ha", {}) or {}
        pix_counts  = getattr(areas, "pixel_counts", {}) or {}
        px_res      = getattr(areas, "pixel_res_m", None)

        if areas_ha:
            header = ["Clase", "Píxeles", "Área (ha)"]
            rows   = [header]
            for cls_id in sorted(areas_ha.keys()):
                rows.append([
                    str(cls_id),
                    str(pix_counts.get(cls_id, "N/D")),
                    f"{areas_ha[cls_id]:.2f}",
                ])
            t = Table(rows, colWidths=[4 * cm, 6 * cm, 6 * cm])
            t.setStyle(_HEADER_STYLE)
            story.append(t)
            total = getattr(areas, "total_area_ha", None)
            res_str = f"{px_res:.2f} m" if px_res is not None else "N/D"
            story.append(Paragraph(
                f"Área total: {total:.2f} ha  |  Resolución espacial: {res_str}/px"
                if total is not None else f"Resolución espacial: {res_str}/px",
                styles["caption"],
            ))


def _section_preprocessing(
    styles: dict,
    session_data: dict,
    story: list,
) -> None:
    story.append(Paragraph("3. Métricas de Preprocesamiento", styles["section_h1"]))

    cfg = session_data.get("config") or {}

    rows = [
        ("CRS objetivo",            str(cfg.get("target_crs",           "N/D"))),
        ("Resolución (m/px)",       str(cfg.get("target_resolution_m",  "N/D"))),
        ("ENL (default SAR)",       str(cfg.get("sar_enl_default",      "N/D"))),
        ("ENL Sentinel-1 IW",       str(cfg.get("sar_enl_sentinel1_iw", "N/D"))),
        ("Overlap mínimo (%)",      str(cfg.get("min_overlap_pct",      "N/D"))),
        ("Max resolución ratio (×)",str(cfg.get("max_resolution_ratio", "N/D"))),
        ("Max gap fechas (días)",   str(cfg.get("max_date_gap_days",    "N/D"))),
        ("OA mínima (gate)",        str(cfg.get("min_oa_threshold",     "N/D"))),
        ("F1 mínimo (gate)",        str(cfg.get("min_minority_f1",      "N/D"))),
    ]
    story.append(_two_col_table(rows))

    # Independent accuracy assessment
    acc = session_data.get("accuracy")
    if acc is not None:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Accuracy Assessment (puntos de referencia)", styles["section_h2"]))
        acc_rows = [
            ("Overall Accuracy",   _g(acc, "oa")),
            ("Cohen's Kappa",      _g(acc, "kappa")),
            ("Puntos totales",     _g(acc, "n_points")),
            ("Puntos válidos",     _g(acc, "n_valid")),
            ("Puntos descartados", _g(acc, "n_discarded")),
        ]
        story.append(_two_col_table(acc_rows))


def _section_audit_log(
    styles: dict,
    run_id: str,
    story: list,
) -> None:
    story.append(Paragraph("4. Log de Auditoría", styles["section_h1"]))

    # Import here to keep pipeline boundary clean (no circular imports)
    try:
        from pipeline import audit as _audit
        events: list[dict] = _audit.get_log(run_id)
    except Exception:
        events = []

    if not events:
        story.append(Paragraph("No se encontraron eventos de auditoría para este run.", styles["body"]))
        return

    truncated = len(events) > MAX_AUDIT_ROWS
    display   = events[-MAX_AUDIT_ROWS:] if truncated else events

    header = ["#", "Tipo", "Stage / Acción", "Decisión", "Timestamp"]
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
            f"(Mostrando los últimos {MAX_AUDIT_ROWS} de {len(events)} eventos. "
            "Consulte el audit_log.json completo para el historial íntegro.)",
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
                    optional; missing values render as "N/D" (No Disponible).
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
    story: list = []

    _section_cover(styles, run_id, operator_name, generated_at, story)

    # Page break after cover
    from reportlab.platypus import PageBreak
    story.append(PageBreak())

    _section_executive_summary(styles, session_data, story)
    story.append(PageBreak())

    _section_classification(styles, session_data, story)
    story.append(PageBreak())

    _section_preprocessing(styles, session_data, story)
    story.append(PageBreak())

    _section_audit_log(styles, run_id, story)

    doc.build(story)
    return out_path.resolve()
