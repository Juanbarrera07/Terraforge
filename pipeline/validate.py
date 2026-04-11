"""
Phase 2B — Validation gates.

Every check returns a ValidationResult (or list thereof).
The run_all_validations() orchestrator collects all results into a flat dict.
Callers use has_critical_failures() to decide whether to block pipeline progression.

Severity levels
---------------
"ok"    — check passed
"warn"  — soft failure; user is warned but may proceed
"error" — hard failure; is_critical=True blocks pipeline progression

Critical checks (hard-block on failure)
----------------------------------------
- crs_match
- minimum_overlap
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from pipeline.raster_io import compute_overlap_pct

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ValidationResult:
    check:       str
    status:      str              # "ok" | "warn" | "error"
    message:     str
    is_critical: bool = False
    detail:      Optional[dict]  = field(default=None, compare=False)

    @property
    def passed(self) -> bool:
        return self.status == "ok"

    @property
    def blocks_pipeline(self) -> bool:
        return self.is_critical and self.status == "error"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _crs_key(crs) -> Optional[str]:
    """Return a stable string key for a CRS object; None if CRS is missing."""
    if crs is None:
        return None
    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        pass
    return f"EPSG:{epsg}" if epsg else crs.to_wkt()[:80]


# ── Individual checks ─────────────────────────────────────────────────────────

def check_crs_match(layers: dict[str, dict]) -> ValidationResult:
    """All layers must share the same CRS."""
    check = "crs_match"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no CRS comparison needed.")

    crs_map = {name: _crs_key(layer["meta"].get("crs")) for name, layer in layers.items()}
    missing  = [k for k, v in crs_map.items() if v is None]
    unique   = {v for v in crs_map.values() if v is not None}

    if missing:
        return ValidationResult(
            check, "error",
            f"Missing CRS on layer(s): {', '.join(missing)}. Assign a CRS before uploading.",
            is_critical=True,
            detail={"crs_map": crs_map},
        )
    if len(unique) > 1:
        return ValidationResult(
            check, "error",
            f"CRS mismatch across layers: {crs_map}. Reproject all layers to a common CRS.",
            is_critical=True,
            detail={"crs_map": crs_map},
        )
    return ValidationResult(
        check, "ok",
        f"All layers share {list(unique)[0]}.",
        detail={"crs_map": crs_map},
    )


def check_resolution_compatibility(
    layers: dict[str, dict],
    max_ratio: float = 2.0,
) -> ValidationResult:
    """Warn if any two layers differ in GSD by more than max_ratio."""
    check = "resolution_compatibility"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no resolution comparison needed.")

    res_map = {
        name: abs(float(layer["meta"]["res"][0]))
        for name, layer in layers.items()
    }
    vals = list(res_map.values())
    min_r, max_r = min(vals), max(vals)
    ratio = max_r / min_r if min_r > 0 else float("inf")

    if ratio > max_ratio:
        return ValidationResult(
            check, "warn",
            (
                f"Resolution ratio {ratio:.1f}× exceeds threshold ({max_ratio}×). "
                f"Coregistration accuracy may be reduced. Values: {res_map}"
            ),
            detail={"res_map": res_map, "ratio": round(ratio, 3)},
        )
    return ValidationResult(
        check, "ok",
        f"Resolution ratio {ratio:.1f}× is within threshold ({max_ratio}×).",
        detail={"res_map": res_map, "ratio": round(ratio, 3)},
    )


def check_date_proximity(
    layers: dict[str, dict],
    max_gap_days: int = 30,
    phenology_month_gap: int = 2,
) -> list[ValidationResult]:
    """
    Return up to two ValidationResults:
      1. date_proximity  — error/warn on acquisition date gap
      2. phenology_risk  — warn if layers span > phenology_month_gap months
    """
    results: list[ValidationResult] = []
    dates: dict[str, date] = {
        name: layer["date"]
        for name, layer in layers.items()
        if layer.get("date") is not None
    }

    # Not enough parseable dates — soft warning
    if len(dates) < 2:
        results.append(ValidationResult(
            "date_proximity", "warn",
            "Acquisition date could not be parsed for all layers. "
            "Verify temporal alignment manually.",
        ))
        return results

    date_vals = list(dates.values())
    max_gap = max(
        abs((a - b).days)
        for i, a in enumerate(date_vals)
        for b in date_vals[i + 1:]
    )

    if max_gap > max_gap_days:
        results.append(ValidationResult(
            "date_proximity", "warn",
            f"Maximum acquisition date gap: {max_gap} days (threshold: {max_gap_days} days). "
            "Large temporal gaps may introduce change artefacts.",
            detail={"dates": {k: str(v) for k, v in dates.items()}, "max_gap_days": max_gap},
        ))
    else:
        results.append(ValidationResult(
            "date_proximity", "ok",
            f"All layers acquired within {max_gap} days of each other.",
            detail={"max_gap_days": max_gap},
        ))

    # Phenological warning
    months = [d.month for d in dates.values()]
    month_span = max(months) - min(months)
    if month_span > phenology_month_gap:
        results.append(ValidationResult(
            "phenology_risk", "warn",
            f"Seasonal variation risk — layers span {month_span} calendar months. "
            "Verify NDVI and vegetation features for phenological bias.",
            detail={"months": months, "span": month_span},
        ))

    return results


def check_minimum_overlap(
    layers: dict[str, dict],
    min_pct: float = 80.0,
) -> ValidationResult:
    """
    Verify pairwise spatial overlap between all layers.
    Overlap is measured as a percentage of the smaller layer's bounding area.

    NOTE: Bounds are assumed to be in the same CRS.  Run check_crs_match first.
    """
    check = "minimum_overlap"
    if len(layers) < 2:
        return ValidationResult(check, "ok", "Single layer — no overlap check needed.")

    names       = list(layers.keys())
    bounds_list = [layers[n]["meta"]["bounds"] for n in names]

    pair_results: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pct = compute_overlap_pct(bounds_list[i], bounds_list[j])
            pair_results.append((names[i], names[j], pct))

    min_overlap = min(p for _, _, p in pair_results)
    failing = [(a, b, f"{p:.1f}%") for a, b, p in pair_results if p < min_pct]

    if failing:
        return ValidationResult(
            check, "error",
            f"Insufficient spatial overlap. Pairs below {min_pct}%: {failing}",
            is_critical=True,
            detail={"pairs": [(a, b, round(p, 2)) for a, b, p in pair_results]},
        )
    return ValidationResult(
        check, "ok",
        f"All layer pairs overlap ≥ {min_overlap:.1f}% (threshold: {min_pct}%).",
        detail={"pairs": [(a, b, round(p, 2)) for a, b, p in pair_results]},
    )


def check_band_counts(layers: dict[str, dict]) -> list[ValidationResult]:
    """
    Validate that band counts are consistent with the declared layer type.
    Returns one ValidationResult per layer.
    """
    results: list[ValidationResult] = []
    for name, layer in layers.items():
        count = layer["meta"]["count"]
        ltype = layer["layer_type"]
        check = f"band_count_{name}"

        if ltype == "sar":
            if count not in (1, 2, 4):
                results.append(ValidationResult(
                    check, "warn",
                    f"SAR layer '{name}' has {count} bands. "
                    "Expected 1 (single-pol), 2 (dual-pol), or 4 (quad-pol).",
                ))
            else:
                results.append(ValidationResult(
                    check, "ok",
                    f"'{name}': {count} SAR band(s) — valid.",
                ))
        elif ltype == "dem":
            if count != 1:
                results.append(ValidationResult(
                    check, "warn",
                    f"DEM layer '{name}' has {count} bands; expected 1.",
                ))
            else:
                results.append(ValidationResult(check, "ok", f"'{name}': 1 DEM band — valid."))
        elif ltype == "optical":
            if count < 3:
                results.append(ValidationResult(
                    check, "error",
                    f"Optical layer '{name}' has only {count} band(s). "
                    "Minimum 3 bands required for spectral analysis.",
                    is_critical=False,
                ))
            else:
                results.append(ValidationResult(
                    check, "ok",
                    f"'{name}': {count} optical band(s) — valid.",
                ))
        else:
            results.append(ValidationResult(
                check, "ok",
                f"'{name}': {count} band(s) — type '{ltype}' (no specific band check).",
            ))

    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_all_validations(
    layers: dict[str, dict],
    config: dict,
) -> dict[str, ValidationResult]:
    """
    Run all validation checks and return a flat dict keyed by check name.

    Execution order matters: CRS check runs before overlap (overlap result is
    only meaningful when CRS is shared).
    """
    results: dict[str, ValidationResult] = {}

    def _add(r: ValidationResult | list[ValidationResult]) -> None:
        if isinstance(r, list):
            for item in r:
                results[item.check] = item
        else:
            results[r.check] = r

    _add(check_crs_match(layers))
    _add(check_resolution_compatibility(
        layers,
        max_ratio=float(config.get("max_resolution_ratio", 2.0)),
    ))
    _add(check_date_proximity(
        layers,
        max_gap_days=int(config.get("max_date_gap_days", 30)),
        phenology_month_gap=int(config.get("phenology_month_gap", 2)),
    ))
    _add(check_minimum_overlap(
        layers,
        min_pct=float(config.get("min_overlap_pct", 80.0)),
    ))
    _add(check_band_counts(layers))

    return results


def has_critical_failures(results: dict[str, ValidationResult]) -> bool:
    """Return True if any result is both critical and errored."""
    return any(r.blocks_pipeline for r in results.values())


def validation_summary(results: dict[str, ValidationResult]) -> dict[str, int]:
    """Return counts of ok / warn / error results."""
    counts = {"ok": 0, "warn": 0, "error": 0}
    for r in results.values():
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts
