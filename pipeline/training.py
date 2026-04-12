"""
Phase 5B — Training data extraction from shapefiles and label rasters.

Provides two extraction paths with a unified return signature, plus a helper
for building the metadata dict consumed by validate.check_shapefile_alignment.

Public API
----------
extract_from_shapefile    — per-polygon windowed extraction via rasterio.mask
extract_from_label_raster — windowed extraction from an aligned label raster
build_shapefile_meta      — build shp_meta dict for validate.check_shapefile_alignment

Design rules
------------
- No Streamlit imports.
- Shapefile extraction uses rasterio.mask per polygon; all_touched=False to
  avoid boundary contamination.  Only the crop bounding box of each polygon
  is read into Python — no full-raster loads.
- extract_from_label_raster produces identical (X, y) to
  classify.extract_training_samples when called with identical inputs.
  (Note: parameter order differs — label_path is first here.)
- All nodata / fill pixels are excluded before any array is built.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as _rio_mask

from pipeline.raster_io import DEFAULT_BLOCK, iter_windows


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_class_summary(y: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame with columns [class, count, percentage]."""
    counts = Counter(y.tolist())
    total  = len(y)
    return pd.DataFrame([
        {
            "class":      int(cls),
            "count":      int(cnt),
            "percentage": round(cnt / total * 100.0, 2),
        }
        for cls, cnt in sorted(counts.items())
    ])


def _stratified_subsample(
    X:            np.ndarray,
    y:            np.ndarray,
    max_samples:  int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified random subsample — proportional allocation per class.

    Each class receives ``round(max_samples * class_pct)`` samples, minimum 1.
    Total may differ from max_samples by at most n_classes due to rounding.
    """
    rng            = np.random.default_rng(random_state)
    classes, cnts  = np.unique(y, return_counts=True)
    n_total        = len(y)
    keep: list[np.ndarray] = []

    for cls, cnt in zip(classes, cnts):
        idx    = np.where(y == cls)[0]
        budget = max(1, round(max_samples * int(cnt) / n_total))
        budget = min(budget, int(cnt))
        keep.append(rng.choice(idx, size=budget, replace=False))

    idx_all = np.concatenate(keep)
    rng.shuffle(idx_all)
    return X[idx_all], y[idx_all]


# ── Public API ────────────────────────────────────────────────────────────────

def extract_from_shapefile(
    shp_path:                str | Path,
    feature_path:            str | Path,
    class_column:            str,
    max_samples_per_polygon: int = 500,
    random_state:            int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract training samples from a shapefile of labelled polygons.

    Each polygon is processed individually via ``rasterio.mask`` with
    ``all_touched=False``.  Only the crop bounding box of each polygon is
    loaded into Python; no full-raster array is ever created.  Pixels inside
    the polygon that are finite and non-nodata are collected as feature vectors.
    A per-polygon sample cap prevents large polygons from dominating the dataset.

    The shapefile is automatically reprojected to the feature stack CRS when
    their CRS objects differ.

    Parameters
    ----------
    shp_path                : Shapefile path (.shp or .zip containing the bundle).
    feature_path            : Feature stack GeoTIFF (from features.compute_features).
    class_column            : Column holding integer class labels (one per polygon).
    max_samples_per_polygon : Maximum pixels sampled from each polygon.  Excess is
                              drawn uniformly without replacement.  0 disables cap.
    random_state            : RNG seed for per-polygon subsampling.

    Returns
    -------
    X             : (n_samples, n_features) float32 — feature vectors.
    y             : (n_samples,)            int32   — class labels.
    class_summary : pd.DataFrame with columns [class, count, percentage].

    Raises
    ------
    ImportError : If geopandas is not installed.
    ValueError  : Missing class column, non-polygon geometry, missing CRS,
                  fewer than 2 unique classes, or no valid samples extracted.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for shapefile-based training.  "
            "Install with: micromamba install -c conda-forge geopandas"
        ) from exc

    shp_path     = Path(shp_path)
    feature_path = Path(feature_path)

    gdf = gpd.read_file(str(shp_path))

    # ── Validate class column ──────────────────────────────────────────────────
    if class_column not in gdf.columns:
        available = [c for c in gdf.columns if c != "geometry"]
        raise ValueError(
            f"Column '{class_column}' not found in shapefile. "
            f"Available columns: {available}"
        )

    # ── Validate geometry type ─────────────────────────────────────────────────
    valid_types = {"Polygon", "MultiPolygon"}
    found_types = set(gdf.geometry.geom_type.unique())
    bad_types   = found_types - valid_types
    if bad_types:
        raise ValueError(
            f"Unsupported geometry type(s): {bad_types}. "
            "Use Polygon or MultiPolygon features only."
        )

    # ── Validate CRS ──────────────────────────────────────────────────────────
    if gdf.crs is None:
        raise ValueError(
            "Shapefile has no CRS.  Assign a coordinate reference system "
            "before using it for training."
        )

    # ── Validate class count ──────────────────────────────────────────────────
    unique_classes = gdf[class_column].dropna().unique()
    if len(unique_classes) < 2:
        raise ValueError(
            f"Only {len(unique_classes)} unique class value(s) found in column "
            f"'{class_column}'.  At least 2 classes are required for training."
        )

    # ── Reproject to feature CRS if needed ────────────────────────────────────
    with rasterio.open(feature_path) as src:
        feat_crs    = src.crs
        feat_nodata = src.nodata
        n_bands     = src.count

    if gdf.crs != feat_crs:
        gdf = gdf.to_crs(feat_crs)

    # ── Per-polygon extraction ─────────────────────────────────────────────────
    fill_val = feat_nodata if feat_nodata is not None else -9999.0
    cap      = int(max_samples_per_polygon) if max_samples_per_polygon else 0
    rng      = np.random.default_rng(random_state)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    with rasterio.open(feature_path) as src:
        for _, row in gdf.iterrows():
            geom      = row.geometry
            class_val = int(row[class_column])

            try:
                masked, _ = _rio_mask(
                    src,
                    [geom],
                    crop        = True,
                    nodata      = fill_val,
                    all_touched = False,
                    filled      = True,
                )
            except Exception:
                # Polygon outside raster extent or zero-area intersection — skip
                continue

            # masked: (n_bands, H, W) — pixels outside polygon filled with fill_val
            valid = np.ones(masked.shape[1:], dtype=bool)
            for b in range(n_bands):
                valid &= np.isfinite(masked[b])
                valid &= masked[b] != fill_val

            if not valid.any():
                continue

            # (n_valid, n_bands) — C-contiguous copy
            X_poly = masked[:, valid].T.astype(np.float32)

            if cap > 0 and X_poly.shape[0] > cap:
                idx    = rng.choice(X_poly.shape[0], size=cap, replace=False)
                X_poly = X_poly[idx]

            X_parts.append(X_poly)
            y_parts.append(np.full(X_poly.shape[0], class_val, dtype=np.int32))

    if not X_parts:
        raise ValueError(
            "No valid samples could be extracted from the shapefile.  "
            "Verify that the polygons overlap the feature raster extent "
            "and that the feature raster has valid pixel values."
        )

    X             = np.concatenate(X_parts, axis=0)
    y             = np.concatenate(y_parts, axis=0)
    class_summary = _compute_class_summary(y)
    return X, y, class_summary


def extract_from_label_raster(
    label_path:   str | Path,
    feature_path: str | Path,
    nodata_label: int           = 0,
    max_samples:  Optional[int] = None,
    random_state: int           = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract training samples from a label raster aligned with the feature stack.

    Produces identical (X, y) arrays to ``classify.extract_training_samples``
    when called with the same input files and ``nodata_label`` / ``random_state``.
    Note the parameter order here differs: ``label_path`` comes first.

    Parameters
    ----------
    label_path   : Single-band integer raster.  Each pixel value is a class label;
                   ``nodata_label`` marks unlabeled pixels.  The file-level nodata
                   is also excluded.
    feature_path : Feature stack GeoTIFF (from features.compute_features).
    nodata_label : Integer pixel value meaning "not labeled".  Default 0.
    max_samples  : If set, stratified-randomly subsample to at most this many
                   samples after collection (proportional per class).
    random_state : RNG seed for optional subsampling.

    Returns
    -------
    X             : (n_samples, n_features) float32 — feature vectors.
    y             : (n_samples,)            int32   — class labels.
    class_summary : pd.DataFrame with columns [class, count, percentage].

    Raises
    ------
    ValueError : Raster shape mismatch or no labeled samples found.
    """
    feature_path = Path(feature_path)
    label_path   = Path(label_path)

    with rasterio.open(feature_path) as fsrc, rasterio.open(label_path) as lsrc:
        if (fsrc.height, fsrc.width) != (lsrc.height, lsrc.width):
            raise ValueError(
                f"Raster shape mismatch: feature ({fsrc.height}×{fsrc.width}) "
                f"vs label ({lsrc.height}×{lsrc.width})."
            )
        n_features  = fsrc.count
        feat_nodata = fsrc.nodata
        lab_nodata  = lsrc.nodata

        X_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []

        for window in iter_windows(fsrc, DEFAULT_BLOCK):
            feat = fsrc.read(window=window).astype(np.float32)  # (n_feat, H, W)
            lab  = lsrc.read(1, window=window)                  # (H, W)

            valid_lab = lab != nodata_label
            if lab_nodata is not None:
                valid_lab &= lab != lab_nodata

            valid_feat = np.ones(lab.shape, dtype=bool)
            for b in range(n_features):
                valid_feat &= np.isfinite(feat[b])
                if feat_nodata is not None:
                    valid_feat &= feat[b] != feat_nodata

            mask = valid_lab & valid_feat
            if not mask.any():
                continue

            X_parts.append(feat[:, mask].T)
            y_parts.append(lab[mask].astype(np.int32))

    if not X_parts:
        raise ValueError(
            f"No labeled training samples found.  "
            f"Verify the label raster has pixel values other than "
            f"nodata_label={nodata_label}."
        )

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    if max_samples is not None and len(y) > max_samples:
        X, y = _stratified_subsample(X, y, max_samples, random_state)

    class_summary = _compute_class_summary(y)
    return X, y, class_summary


def build_shapefile_meta(
    shp_path:     str | Path,
    class_column: str,
    feature_crs,              # rasterio.crs.CRS
) -> dict:
    """
    Build the ``shp_meta`` dict consumed by
    ``validate.check_shapefile_alignment``.

    Reads the shapefile header (no pixel data) and reprojects to
    ``feature_crs`` to compute the bounding box in the raster coordinate system.

    Parameters
    ----------
    shp_path     : Shapefile path (.shp or .zip).
    class_column : Column to read class labels from; if absent, class_labels
                   is returned as an empty list.
    feature_crs  : CRS of the feature stack (rasterio CRS object).

    Returns
    -------
    dict with keys:
        ``"crs"``          — original CRS of the shapefile (may be None)
        ``"bounds"``       — ``rasterio.coords.BoundingBox`` in feature CRS
        ``"class_labels"`` — sorted list of unique integer class labels
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required for build_shapefile_meta.") from exc

    from rasterio.coords import BoundingBox

    gdf          = gpd.read_file(str(Path(shp_path)))
    original_crs = gdf.crs

    if original_crs is not None and original_crs != feature_crs:
        gdf_proj = gdf.to_crs(feature_crs)
    else:
        gdf_proj = gdf

    b = gdf_proj.total_bounds          # [minx, miny, maxx, maxy]
    bounds = BoundingBox(
        left   = float(b[0]),
        bottom = float(b[1]),
        right  = float(b[2]),
        top    = float(b[3]),
    )

    class_labels: list[int] = []
    if class_column in gdf.columns:
        try:
            class_labels = sorted(
                int(v) for v in gdf[class_column].dropna().unique()
            )
        except (ValueError, TypeError):
            class_labels = []

    return {
        "crs":          original_crs,
        "bounds":       bounds,
        "class_labels": class_labels,
    }
