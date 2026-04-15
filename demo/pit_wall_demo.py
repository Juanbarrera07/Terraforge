#\!/usr/bin/env python3
"""
TerraForge Mining Intelligence — End-to-End Classification Demo
================================================================
Runs the full TerraForge pipeline on open satellite or drone data and
produces a classified COG, confidence raster, STAC metadata, audit log,
PDF report, and a ZIP archive — all in one command.

Supported modes
---------------
  pit_wall  : 5-class pit wall classification (Sentinel-2 or drone RGB)
  tsf       : 6-class tailings storage facility classification (S2 + optional SAR)
  both      : run pit_wall then tsf sequentially

Usage
-----
  python demo/pit_wall_demo.py \
      --site tom_price \
      --mode pit_wall \
      --data-dir demo/open_data/ \
      --output-dir demo/outputs/

  # Drone data (DJI Air 3S RGB):
  python demo/pit_wall_demo.py \
      --site custom \
      --mode pit_wall \
      --data-dir /path/to/drone/tiles/ \
      --output-dir demo/outputs/ \
      --drone

  # Custom labels:
  python demo/pit_wall_demo.py \
      --site tom_price \
      --mode pit_wall \
      --data-dir demo/open_data/ \
      --output-dir demo/outputs/ \
      --labels /path/to/labels.tif
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline.config_loader import load_config
from pipeline.ingest import ingest_path
from pipeline.validate import run_all_validations, check_drone_inputs
from pipeline.preprocess import dos1_atmospheric_correction
from pipeline.features import BandMap, compute_features, active_features
from pipeline.classify import (
    ClassificationConfig,
    extract_training_samples,
    train_model,
    predict_raster,
)
from pipeline.postprocess import run_postprocess_chain, compute_class_areas
from pipeline.export import (
    write_cog,
    build_stac_item,
    write_stac_item,
    export_audit_log,
    package_run,
)
from pipeline.audit import log_event
from pipeline.report import generate_report

# ── Site presets ──────────────────────────────────────────────────────────────

SITE_PRESETS: dict[str, dict] = {
    "tom_price": {
        "name": "Mount Tom Price Iron Ore Mine",
        "operator": "Rio Tinto",
        "state": "WA",
        "lat": -22.6931,
        "lon": 117.7939,
    },
    "boddington": {
        "name": "Boddington Gold Mine",
        "operator": "Newmont",
        "state": "WA",
        "lat": -32.7996,
        "lon": 116.3748,
    },
    "super_pit": {
        "name": "Fimiston Open Pit (Super Pit)",
        "operator": "Northern Star / Saracen JV",
        "state": "WA",
        "lat": -30.7775,
        "lon": 121.5107,
    },
    "custom": {
        "name": "Custom Site",
        "operator": "N/A",
        "state": "N/A",
        "lat": 0.0,
        "lon": 0.0,
    },
}

# ── Band map builders ─────────────────────────────────────────────────────────

def _build_band_map_satellite(mode: str, has_sar: bool = False) -> BandMap:
    """
    Band map for a 7-band Sentinel-2 L2A stack
    (B02/B03/B04/B8A/B11/B12 resampled to 10 m + optional SAR).
    Band order matches the gdalbuildvrt stack in open_data/README.md:
        1=Blue(B02) 2=Green(B03) 3=Red(B04) 4=RedEdge(B8A) 5=NIR(B08 or B8A)
        6=SWIR1(B11) 7=SWIR2(B12)  [8=VV  9=VH if SAR appended]
    Adjust if your stack order differs.
    """
    if has_sar:
        return BandMap(
            blue=1, green=2, red=3, rededge=4, nir=5, swir=6,
            vv=8, vh=9,
        )
    return BandMap(blue=1, green=2, red=3, rededge=4, nir=5, swir=6)


def _build_band_map_drone() -> BandMap:
    """RGB-only band map for DJI Air 3S imagery (Red=1, Green=2, Blue=3)."""
    return BandMap(red=1, green=2, blue=3)


# ── Label helpers ─────────────────────────────────────────────────────────────

def _generate_synthetic_labels(
    src_raster: Path,
    out_path: Path,
    mode: str,
) -> Path:
    """
    Call create_demo_labels.py to produce a geotechnically coherent label raster
    aligned to *src_raster*.  Falls back to a simple NumPy label raster if the
    script is not found.
    """
    labels_script = _REPO_ROOT / "demo" / "labels" / "create_demo_labels.py"
    if labels_script.exists():
        import subprocess
        cmd = [
            sys.executable, str(labels_script),
            "--src", str(src_raster),
            "--out", str(out_path),
            "--mode", mode,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists():
            return out_path
        print(f"  [warn] create_demo_labels.py failed: {result.stderr.strip()}")

    # Fallback: checkerboard label raster using rasterio
    _generate_fallback_labels(src_raster, out_path, mode)
    return out_path


def _generate_fallback_labels(src_raster: Path, out_path: Path, mode: str) -> None:
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    n_classes = 5 if mode == "pit_wall" else 6

    with rasterio.open(src_raster) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()

    rng = np.random.default_rng(42)
    labels = np.zeros((H, W), dtype=np.int16)
    # Simple radial assignment
    cy, cx = H // 2, W // 2
    max_r = min(H, W) // 2
    yy, xx = np.mgrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    thresholds = np.linspace(0, max_r, n_classes + 1)
    for i in range(n_classes):
        mask = (r >= thresholds[i]) & (r < thresholds[i + 1])
        labels[mask] = i + 1
    # Add noise
    noise_mask = rng.random((H, W)) < 0.05
    labels[noise_mask] = rng.integers(1, n_classes + 1, noise_mask.sum())

    profile.update(count=1, dtype="int16", nodata=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labels[np.newaxis])


# ── Progress helpers ──────────────────────────────────────────────────────────

def _bar(label: str, current: int, total: int) -> None:
    pct = int(current / max(total, 1) * 40)
    bar = "#" * pct + "-" * (40 - pct)
    print(f"\r  [{bar}] {current}/{total}  {label}    ", end="", flush=True)


def _make_progress(label: str):
    def cb(cur, tot):
        _bar(label, cur, tot)
    return cb


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _step(msg: str) -> None:
    print(f"  >> {msg}")


# ── Core pipeline runner ──────────────────────────────────────────────────────

def run_pipeline(
    mode: str,
    data_dir: Path,
    output_dir: Path,
    site: str,
    labels_path: Path | None,
    drone: bool,
    cfg: dict,
) -> dict:
    """
    Run the full TerraForge pipeline for one mode (pit_wall or tsf).
    Returns a summary dict with metrics and output paths.
    """
    t0 = time.perf_counter()
    run_id = uuid.uuid4().hex[:8].upper()
    run_out = output_dir / run_id
    run_out.mkdir(parents=True, exist_ok=True)

    site_info = SITE_PRESETS.get(site, SITE_PRESETS["custom"])
    class_defs = (
        cfg.get("pit_wall_classes", {}) if mode == "pit_wall"
        else cfg.get("tsf_classes", {})
    )

    _section(f"RUN {run_id}  |  {site_info['name']}  |  {mode.upper()}")
    log_event(run_id, "run_start", {
        "mode": mode, "site": site, "drone": drone,
        "data_dir": str(data_dir),
    })

    # ── 1. Discover input rasters ─────────────────────────────────────────────
    _section("STEP 1 — Ingest")
    tif_files = sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))
    if not tif_files:
        raise FileNotFoundError(
            f"No .tif files found in {data_dir}. "
            f"See demo/open_data/README.md for download instructions."
        )
    _step(f"Found {len(tif_files)} raster file(s)")

    # Pick the primary stack: prefer *_stack* files, else largest .tif
    stack_candidates = [f for f in tif_files if "stack" in f.stem.lower()]
    primary_tif = stack_candidates[0] if stack_candidates else max(tif_files, key=lambda p: p.stat().st_size)
    _step(f"Primary raster: {primary_tif.name}")

    layers: dict = {}
    for tif in tif_files:
        try:
            layer = ingest_path(tif)
            layers[tif.stem] = layer
            log_event(run_id, "ingestion", {"file": tif.name, "sensor": layer["sensor"]})
        except Exception as exc:
            _step(f"  [warn] Could not ingest {tif.name}: {exc}")

    # ── 2. Validate ───────────────────────────────────────────────────────────
    _section("STEP 2 — Validate")
    if drone:
        drone_results = check_drone_inputs(primary_tif, cfg)
        for vr in drone_results:
            symbol = "PASS" if vr.passed else ("WARN" if vr.level == "warning" else "FAIL")
            _step(f"  [{symbol}] {vr.check}: {vr.message}")
            log_event(run_id, "gate", {
                "check": vr.check,
                "passed": vr.passed,
                "level": vr.level,
                "message": vr.message,
            }, decision="proceed" if vr.passed or vr.level == "warning" else "block")

    if len(layers) > 1:
        val_results = run_all_validations(layers, cfg)
        for key, vr in val_results.items():
            symbol = "PASS" if vr.passed else ("WARN" if vr.level == "warning" else "FAIL")
            _step(f"  [{symbol}] {key}: {vr.message}")
    else:
        _step("  Single input — skipping multi-layer validations")

    # ── 3. Preprocess ─────────────────────────────────────────────────────────
    _section("STEP 3 — Preprocess")
    preprocessed_path = run_out / "preprocessed.tif"

    if drone:
        # Drone RGB: copy as-is (scaling handled inside compute_features)
        import shutil
        shutil.copy2(primary_tif, preprocessed_path)
        _step(f"Drone mode: copied {primary_tif.name} -> preprocessed.tif")
        _step("  Note: uint8 -> float32 scaling applied during feature computation")
    else:
        _step("Applying DOS1 atmospheric correction ...")
        dos1_atmospheric_correction(
            primary_tif,
            preprocessed_path,
            progress=_make_progress("DOS1"),
        )
        print()  # newline after progress bar
        _step(f"Wrote preprocessed.tif ({preprocessed_path.stat().st_size / 1e6:.1f} MB)")

    log_event(run_id, "decision", {"stage": "preprocess", "input": primary_tif.name})

    # ── 4. Feature engineering ────────────────────────────────────────────────
    _section("STEP 4 — Features")
    has_sar = mode == "tsf" and not drone and any("S1" in f.stem for f in tif_files)
    band_map = _build_band_map_drone() if drone else _build_band_map_satellite(mode, has_sar)

    feat_list = active_features(band_map, cfg)
    _step(f"Band map: {[k for k, v in band_map.__dict__.items() if v is not None]}")
    _step(f"Features ({len(feat_list)}): {', '.join(feat_list)}")

    features_path = run_out / "features.tif"
    _step("Computing feature stack ...")
    feat_result = compute_features(
        preprocessed_path,
        features_path,
        band_map,
        cfg=cfg,
        progress=_make_progress("features"),
    )
    print()
    _step(f"Feature stack: {features_path.stat().st_size / 1e6:.1f} MB  |  {len(feat_result.feature_names)} bands")

    if feat_result.high_correlation_pairs:
        _step(f"  [warn] High-correlation pairs: {feat_result.high_correlation_pairs[:3]}")

    log_event(run_id, "gate", {
        "stage": "features",
        "n_features": len(feat_result.feature_names),
        "high_corr_pairs": len(feat_result.high_correlation_pairs),
    }, decision="proceed")

    # ── 5. Labels ─────────────────────────────────────────────────────────────
    _section("STEP 5 — Training Labels")
    if labels_path and labels_path.exists():
        label_tif = labels_path
        _step(f"Using provided labels: {label_tif}")
    else:
        label_tif = run_out / "labels.tif"
        _step(f"Generating synthetic {mode} labels ...")
        _generate_synthetic_labels(preprocessed_path, label_tif, mode)
        _step(f"Synthetic labels written: {label_tif}")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    _section("STEP 6 — Train & Validate")
    _step("Extracting training samples ...")
    X, y = extract_training_samples(features_path, label_tif)
    from collections import Counter
    class_counts = Counter(int(v) for v in y)
    for cls, cnt in sorted(class_counts.items()):
        label = class_defs.get(str(cls), class_defs.get(cls, f"Class {cls}"))
        _step(f"  Class {cls} ({label}): {cnt:,} samples")

    class_cfg = ClassificationConfig(model_type="random_forest")
    _step("Training Random Forest classifier with k-fold CV ...")
    clf_result = train_model(X, y, class_cfg, cfg=cfg, feature_names=feat_result.feature_names)

    gate_sym = "PASS" if clf_result.gate_passed else "FAIL"
    _step(f"  [{gate_sym}] {clf_result.gate_message}")
    _step(f"  OA: {clf_result.oa:.4f}   Kappa: {clf_result.kappa:.4f}   minority-F1: {clf_result.minority_f1:.4f}")
    _step(f"  CV folds OA: {[f'{v:.3f}' for v in clf_result.cv_scores]}")
    _step(f"  SMOTE applied: {clf_result.smote_applied}")

    log_event(run_id, "gate", {
        "stage": "classification",
        "oa": clf_result.oa,
        "kappa": clf_result.kappa,
        "minority_f1": clf_result.minority_f1,
        "gate_passed": clf_result.gate_passed,
        "message": clf_result.gate_message,
    }, decision="proceed" if clf_result.gate_passed else "warn_proceed")

    # ── 7. Predict ────────────────────────────────────────────────────────────
    _section("STEP 7 — Predict Raster")
    classified_raw = run_out / "classified_raw.tif"
    confidence_raw = run_out / "confidence_raw.tif"
    _step("Running tile-by-tile prediction ...")
    predict_raster(
        clf_result.model,
        features_path,
        classified_raw,
        clf_result.feature_names,
        confidence_path=confidence_raw,
    )
    _step(f"Classified raster: {classified_raw.stat().st_size / 1e6:.1f} MB")

    # ── 8. Post-process ───────────────────────────────────────────────────────
    _section("STEP 8 — Post-process")
    import rasterio as _rio
    with _rio.open(preprocessed_path) as ds:
        pixel_res_m = float(abs(ds.transform.a)) if ds.crs and ds.crs.is_projected else 10.0

    _step(f"Pixel resolution: {pixel_res_m:.2f} m  |  drone_mode={drone}")
    chain_results = run_postprocess_chain(
        classified_raw,
        confidence_raw,
        cfg,
        run_out,
        run_id,
        progress=lambda msg: _step(f"  {msg}"),
        pixel_res_m=pixel_res_m,
        drone_mode=drone,
    )
    final_classified = chain_results["final"]
    _step(f"Final classified raster: {Path(final_classified).name}")

    # ── 9. Class areas ────────────────────────────────────────────────────────
    _section("STEP 9 — Area Statistics")
    area_result = compute_class_areas(final_classified)
    total_ha = sum(area_result.areas_ha.values())
    _step(f"Total mapped area: {total_ha:.2f} ha")
    for cls_id, ha in sorted(area_result.areas_ha.items()):
        pct = area_result.areas_pct.get(cls_id, 0.0)
        label = class_defs.get(str(cls_id), class_defs.get(cls_id, f"Class {cls_id}"))
        _step(f"  Class {cls_id} ({label}): {ha:.2f} ha  ({pct:.1f}%)")

    # ── 10. Export COGs + metadata ────────────────────────────────────────────
    _section("STEP 10 — Export")
    cog_path = run_out / f"{run_id}_classified_cog.tif"
    conf_cog = run_out / f"{run_id}_confidence_cog.tif"
    _step("Writing Cloud Optimized GeoTIFFs ...")
    try:
        write_cog(final_classified, cog_path, resampling="nearest")
        write_cog(confidence_raw, conf_cog, resampling="average")
        _step(f"  {cog_path.name}  ({cog_path.stat().st_size / 1e6:.1f} MB)")
        _step(f"  {conf_cog.name}  ({conf_cog.stat().st_size / 1e6:.1f} MB)")
    except RuntimeError as exc:
        _step(f"  [warn] COG export skipped (GDAL not available): {exc}")
        import shutil
        shutil.copy2(final_classified, cog_path)
        shutil.copy2(confidence_raw, conf_cog)

    # STAC item
    stac_path = run_out / f"{run_id}_stac_item.json"
    stac_item = build_stac_item(cog_path, run_id, properties={
        "site": site_info["name"],
        "operator": site_info["operator"],
        "mode": mode,
        "oa": clf_result.oa,
        "kappa": clf_result.kappa,
    })
    write_stac_item(stac_item, stac_path)
    _step(f"  {stac_path.name}")

    # Audit log
    audit_path = run_out / f"{run_id}_audit_log.json"
    export_audit_log(run_id, audit_path)
    _step(f"  {audit_path.name}  ({audit_path.stat().st_size / 1e3:.0f} KB)")

    # ── 11. PDF Report ────────────────────────────────────────────────────────
    _section("STEP 11 — PDF Report")
    report_path = run_out / f"{run_id}_report.pdf"
    try:
        survey_type = "UAV" if drone else "Satellite"
        generate_report(
            run_id=run_id,
            out_path=report_path,
            site_name=site_info["name"],
            operator=site_info["operator"],
            survey_type=survey_type,
            mode=mode,
            cog_path=cog_path,
            confidence_path=conf_cog,
            classification_result=clf_result,
            area_result=area_result,
            class_defs=class_defs,
            cfg=cfg,
            audit_path=audit_path,
        )
        _step(f"PDF report: {report_path.name}  ({report_path.stat().st_size / 1e3:.0f} KB)")
    except Exception as exc:
        _step(f"  [warn] PDF generation failed: {exc}")
        import traceback
        traceback.print_exc()

    # ── 12. ZIP archive ───────────────────────────────────────────────────────
    _section("STEP 12 — Package")
    artifacts = [p for p in [cog_path, conf_cog, stac_path, audit_path, report_path] if p.exists()]
    try:
        manifest = package_run(run_id, artifacts, run_out)
        _step(f"Archive: {manifest.zip_path.name}  ({manifest.zip_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        _step(f"  [warn] Packaging failed: {exc}")
        manifest = None

    # ── 13. Summary ───────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    _section("SUMMARY")
    print(f"  Run ID      : {run_id}")
    print(f"  Site        : {site_info['name']}")
    print(f"  Mode        : {mode}  |  Survey: {'UAV drone' if drone else 'Satellite'}")
    print(f"  OA          : {clf_result.oa:.4f}")
    print(f"  Kappa       : {clf_result.kappa:.4f}")
    print(f"  Minority F1 : {clf_result.minority_f1:.4f}")
    print(f"  Gate        : {'PASS' if clf_result.gate_passed else 'FAIL'}")
    print()
    print(f"  {'Class':<4}  {'Name':<35}  {'ha':>8}  {'%':>6}")
    print(f"  {'-'*4}  {'-'*35}  {'-'*8}  {'-'*6}")
    for cls_id, ha in sorted(area_result.areas_ha.items()):
        pct = area_result.areas_pct.get(cls_id, 0.0)
        label = class_defs.get(str(cls_id), class_defs.get(cls_id, f"Class {cls_id}"))
        print(f"  {cls_id:<4}  {str(label):<35}  {ha:>8.2f}  {pct:>5.1f}%")
    print()
    print(f"  Output directory: {run_out}")
    print()
    for p in artifacts:
        print(f"  {p.stat().st_size/1e3:>8.0f} KB  {p.name}")
    print()
    print(f"  Total time: {elapsed:.1f} s")

    return {
        "run_id": run_id,
        "oa": clf_result.oa,
        "kappa": clf_result.kappa,
        "minority_f1": clf_result.minority_f1,
        "gate_passed": clf_result.gate_passed,
        "areas_ha": area_result.areas_ha,
        "areas_pct": area_result.areas_pct,
        "cog_path": cog_path,
        "report_path": report_path if report_path.exists() else None,
        "zip_path": manifest.zip_path if manifest else None,
        "elapsed_s": elapsed,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pit_wall_demo.py",
        description="TerraForge Mining Intelligence — end-to-end classification demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--site",
        choices=list(SITE_PRESETS.keys()),
        default="tom_price",
        help="Site preset (controls report metadata). Default: tom_price",
    )
    p.add_argument(
        "--mode",
        choices=["pit_wall", "tsf", "both"],
        default="pit_wall",
        help="Classification target. Default: pit_wall",
    )
    p.add_argument(
        "--data-dir",
        default="demo/open_data",
        help="Directory containing input GeoTIFF files. Default: demo/open_data",
    )
    p.add_argument(
        "--output-dir",
        default="demo/outputs",
        help="Root output directory. A subdirectory per run_id is created inside. Default: demo/outputs",
    )
    p.add_argument(
        "--drone",
        action="store_true",
        help="Activate drone mode: RGB-only band map, adaptive post-processing, no atmospheric correction.",
    )
    p.add_argument(
        "--labels",
        default=None,
        help="Path to a label raster (.tif) to use instead of synthetic labels.",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to a custom pipeline_config.yaml (default: config/pipeline_config.yaml).",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"ERROR: data-dir not found: {data_dir}")
        print("  Run from the repo root, e.g.:")
        print("  python demo/pit_wall_demo.py --data-dir demo/open_data/")
        sys.exit(1)

    labels_path: Path | None = None
    if args.labels:
        labels_path = Path(args.labels).resolve()
        if not labels_path.exists():
            print(f"ERROR: --labels file not found: {labels_path}")
            sys.exit(1)

    cfg_path = Path(args.config).resolve() if args.config else None
    cfg = load_config(cfg_path) if cfg_path else load_config()
    # Force mining features on for the demo
    cfg["mining_features_enabled"] = True

    modes = ["pit_wall", "tsf"] if args.mode == "both" else [args.mode]
    results = []
    for m in modes:
        try:
            result = run_pipeline(
                mode=m,
                data_dir=data_dir,
                output_dir=output_dir,
                site=args.site,
                labels_path=labels_path,
                drone=args.drone,
                cfg=cfg,
            )
            results.append(result)
        except FileNotFoundError as exc:
            print(f"\nERROR: {exc}")
            print("\nTo download open data, follow the instructions in:")
            print("  demo/open_data/README.md")
            sys.exit(1)
        except Exception as exc:
            import traceback
            print(f"\nERROR in {m} pipeline: {exc}")
            traceback.print_exc()
            sys.exit(1)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("  COMBINED RUN COMPLETE")
        print("=" * 60)
        for r, m in zip(results, modes):
            print(f"  {m:<10}  OA={r['oa']:.4f}  Kappa={r['kappa']:.4f}  "
                  f"Gate={'PASS' if r['gate_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
