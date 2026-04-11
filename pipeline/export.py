"""
Phase 7 — Export system.

Converts pipeline outputs to Cloud Optimized GeoTIFF (COG), generates STAC 1.0
item JSON, copies the run audit log, and packages everything into a ZIP archive
with SHA-256 checksums.

Public API
----------
ExportManifest   — frozen dataclass describing every artifact in the export
write_cog        — convert a raster to COG (GDAL BuildOverviews + CreateCopy)
build_stac_item  — construct a STAC 1.0 Item dict in memory
write_stac_item  — serialise a STAC item dict to JSON
export_audit_log — copy the run audit log to an export location
package_run      — assemble artifacts into a ZIP with checksums

Design rules
------------
- No Streamlit imports.
- write_cog is a documented exception to the windowed-I/O rule: GDAL overview
  computation and CreateCopy operate on the full band in GDAL-managed memory.
  Memory is bounded by the overview pyramid (not the full raster), but callers
  should be aware that very large rasters will require proportional RAM.
- No rio-cogeo or pystac dependency: GDAL's COPY_SRC_OVERVIEWS path and a
  hand-assembled STAC dict are sufficient for 1.0 compliance.
- package_run discovers artifacts dynamically — no hardcoded filenames.
- SHA-256 checksums are computed from on-disk files before zipping.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import rasterio

try:
    from osgeo import gdal
    gdal.UseExceptions()
    _GDAL_AVAILABLE = True
except ImportError:          # pragma: no cover
    _GDAL_AVAILABLE = False

try:
    from pyproj import Transformer
    _PYPROJ_AVAILABLE = True
except ImportError:          # pragma: no cover
    _PYPROJ_AVAILABLE = False

from pipeline.audit import get_log


# ── Nodata conventions (mirrors raster_io write_raster pattern) ───────────────

_FLOAT_NODATA = -9999.0
_INT_NODATA   = -1

# ── COG creation options ──────────────────────────────────────────────────────

_COG_OVERVIEW_LEVELS = [2, 4, 8, 16]

_COG_OPTIONS_INT = [
    "COMPRESS=DEFLATE",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "COPY_SRC_OVERVIEWS=YES",
    "PREDICTOR=2",
]

_COG_OPTIONS_FLOAT = [
    "COMPRESS=DEFLATE",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "COPY_SRC_OVERVIEWS=YES",
    "PREDICTOR=3",
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExportManifest:
    """
    Describes every artifact produced by package_run().

    Attributes
    ----------
    run_id          : run identifier (8-char hex, uppercase)
    exported_at     : ISO 8601 UTC timestamp of the export
    cog_path        : path to the primary classified COG
    stac_item_path  : path to the STAC item JSON
    audit_log_path  : path to the exported audit log JSON
    zip_path        : path to the ZIP archive
    file_checksums  : {filename: sha256_hex} for every file in the ZIP
    file_sizes      : {filename: byte_count} for every file in the ZIP
    """
    run_id:          str
    exported_at:     str
    cog_path:        Path
    stac_item_path:  Path
    audit_log_path:  Path
    zip_path:        Path
    file_checksums:  dict[str, str]  = field(compare=False)
    file_sizes:      dict[str, int]  = field(compare=False)


# ── Public functions ──────────────────────────────────────────────────────────

def write_cog(
    src_path:   str | Path,
    out_path:   str | Path,
    resampling: str = "nearest",
) -> Path:
    """
    Convert a raster to a Cloud Optimized GeoTIFF.

    .. warning::
        **Documented exception to the windowed-I/O rule.**
        GDAL's ``BuildOverviews`` and ``CreateCopy`` with
        ``COPY_SRC_OVERVIEWS=YES`` operate on the full raster band in
        GDAL-managed memory.  Memory usage scales with the overview pyramid
        (roughly 1/3 of the full raster), not the full raster twice, but
        callers must ensure the raster fits within available RAM.

    Nodata handling
    ---------------
    The source nodata value is preserved if present and finite.
    If the source has no nodata (or NaN as nodata):
    - float32 rasters → nodata = -9999.0
    - integer rasters → nodata = -1

    Parameters
    ----------
    src_path   : source GeoTIFF path.
    out_path   : destination COG path.
    resampling : overview resampling algorithm name passed to BuildOverviews.
                 Common values: "nearest" (classification), "average" (continuous).

    Returns
    -------
    Path to the written COG.
    """
    if not _GDAL_AVAILABLE:
        raise RuntimeError("GDAL (osgeo) is required for write_cog.")

    src_path = Path(src_path)
    out_path = Path(out_path)

    # ── 1. Determine nodata ───────────────────────────────────────────────────
    with rasterio.open(src_path) as ds:
        src_nodata = ds.nodata
        dtype      = ds.dtypes[0]

    is_float = dtype.startswith("float")
    if src_nodata is None or (hasattr(src_nodata, "__float__") and
                               __import__("math").isnan(float(src_nodata))):
        nodata = _FLOAT_NODATA if is_float else float(_INT_NODATA)
    else:
        nodata = float(src_nodata)

    # ── 2. Open source, write nodata if absent, build overviews ──────────────
    src_ds = gdal.Open(str(src_path), gdal.GA_ReadOnly)

    # Write the nodata value into a temporary copy only if the source lacks one,
    # so BuildOverviews uses the correct mask.  If the source already carries it
    # we build overviews directly on the source (read-only is fine for that).
    if src_ds.GetRasterBand(1).GetNoDataValue() is None:
        # Stamp nodata into a /vsimem/ scratch raster so the original is untouched
        mem_driver = gdal.GetDriverByName("GTiff")
        tmp_path_str = f"/vsimem/{src_path.stem}_nd_tmp.tif"
        tmp_ds = mem_driver.CreateCopy(tmp_path_str, src_ds)
        tmp_ds.GetRasterBand(1).SetNoDataValue(nodata)
        tmp_ds.FlushCache()
        src_ds = None
        src_ds = gdal.Open(tmp_path_str, gdal.GA_Update)
    else:
        tmp_path_str = None

    # Build overview pyramid
    src_ds.BuildOverviews(resampling.upper(), _COG_OVERVIEW_LEVELS)
    src_ds.FlushCache()

    # ── 3. CreateCopy to COG ──────────────────────────────────────────────────
    cog_options = _COG_OPTIONS_FLOAT if is_float else _COG_OPTIONS_INT
    driver      = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(
        str(out_path),
        src_ds,
        strict=0,
        options=cog_options,
    )
    # Ensure nodata is written explicitly in the output profile
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None

    # Clean up /vsimem/ scratch if used
    if tmp_path_str:
        gdal.Unlink(tmp_path_str)

    return out_path


def build_stac_item(
    cog_path:   str | Path,
    run_id:     str,
    properties: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Construct a STAC 1.0 Item dict from a COG path and run metadata.

    Bounds are reprojected to EPSG:4326 (WGS-84) via pyproj.
    All coordinate values are native Python ``float`` (not numpy scalars).

    STAC extensions
    ---------------
    The item includes ``proj:epsg`` and ``proj:shape`` from the ``projection``
    extension (https://stac-extensions.github.io/projection/v1.0.0/schema.json).
    Pass ``properties={"classification:classes": [...]}`` to attach optional
    classification or feature metadata.

    Parameters
    ----------
    cog_path   : path to the COG that this item describes.
    run_id     : pipeline run identifier used as part of the item ID.
    properties : extra STAC properties merged into the item.  Keys must be
                 JSON-serialisable.  Reserved keys (datetime, created, run_id,
                 proj:epsg, proj:shape) are set by this function and will
                 override caller-supplied values.

    Returns
    -------
    A plain dict conforming to the STAC 1.0 Item specification.
    """
    if not _PYPROJ_AVAILABLE:
        raise ImportError("pyproj is required for build_stac_item.")

    cog_path = Path(cog_path)

    with rasterio.open(cog_path) as ds:
        bounds   = ds.bounds          # in raster CRS
        crs      = ds.crs
        height   = ds.height
        width    = ds.width
        nodata   = ds.nodata
        dtype    = str(ds.dtypes[0])

    # ── Reproject bounds to WGS-84 ────────────────────────────────────────────
    if crs is not None and crs.to_epsg() != 4326:
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        west,  south = tr.transform(bounds.left,  bounds.bottom)
        east,  north = tr.transform(bounds.right, bounds.top)
    else:
        west, south, east, north = (
            bounds.left, bounds.bottom, bounds.right, bounds.top
        )

    # All coordinates as native Python float (not numpy scalars)
    west, south, east, north = (
        float(west), float(south), float(east), float(north)
    )

    bbox = [west, south, east, north]

    # Closed Polygon ring: 5 coordinate pairs (first == last)
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }

    epsg = int(crs.to_epsg()) if (crs is not None and crs.to_epsg()) else None

    now_utc = datetime.now(timezone.utc).isoformat()

    base_properties: dict[str, Any] = {
        "datetime":   now_utc,
        "created":    now_utc,
        "run_id":     run_id,
        "proj:epsg":  epsg,
        "proj:shape": [height, width],   # [rows, cols] per projection extension
    }
    if nodata is not None:
        base_properties["nodata"] = float(nodata)
    if dtype:
        base_properties["dtype"] = dtype

    # Caller-supplied properties are merged first so reserved keys take priority
    merged_properties = {**(properties or {}), **base_properties}

    item: dict[str, Any] = {
        "type":        "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/projection/v1.0.0/schema.json",
        ],
        "id":       f"{run_id}_{cog_path.stem}",
        "geometry": geometry,
        "bbox":     bbox,
        "properties": merged_properties,
        "links":    [],
        "assets": {
            "data": {
                "href":  cog_path.name,
                "type":  "image/tiff; application=geotiff; profile=cloud-optimized",
                "roles": ["data"],
                "title": cog_path.stem,
            }
        },
    }
    return item


def write_stac_item(
    item_dict: dict[str, Any],
    out_path:  str | Path,
) -> Path:
    """
    Serialise a STAC item dict to a JSON file.

    Parameters
    ----------
    item_dict : dict as returned by build_stac_item().
    out_path  : destination path for the JSON file.
    """
    out_path = Path(out_path)
    out_path.write_text(
        json.dumps(item_dict, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def export_audit_log(
    run_id:   str,
    out_path: str | Path,
) -> Path:
    """
    Copy the run's audit log to an export location.

    Reads from the canonical log store via ``audit.get_log()`` and writes a
    standalone JSON file.  The copy is self-contained: no reference back to the
    ``logs/`` directory is needed to interpret it.

    Parameters
    ----------
    run_id   : run identifier whose log should be exported.
    out_path : destination path for the exported JSON.
    """
    out_path = Path(out_path)
    log      = get_log(run_id)
    out_path.write_text(
        json.dumps(log, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


def package_run(
    run_id:          str,
    artifacts:       list[str | Path],
    out_dir:         str | Path,
) -> ExportManifest:
    """
    Assemble pipeline artifacts into a single ZIP archive with checksums.

    Artifacts are discovered dynamically from the ``artifacts`` list — no
    filenames are hardcoded.  All ``.tif`` and ``.json`` files supplied are
    included; callers build the list from whatever was produced this run.

    SHA-256 checksums and byte sizes are computed from the on-disk files
    *before* zipping.

    Parameters
    ----------
    run_id    : pipeline run identifier (used for the ZIP name).
    artifacts : list of file paths to include (typically: COG, STAC JSON,
                audit log JSON, and any additional per-run outputs).
    out_dir   : directory in which to write ``{run_id}_export.zip``.

    Returns
    -------
    ExportManifest with paths, checksums, and file sizes.

    Raises
    ------
    ValueError  if any path in ``artifacts`` does not exist.
    """
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{run_id}_export.zip"

    paths = [Path(p) for p in artifacts]
    for p in paths:
        if not p.exists():
            raise ValueError(f"Artifact not found: {p}")

    checksums: dict[str, str] = {}
    sizes:     dict[str, int] = {}

    # ── Compute checksums + sizes before zipping ──────────────────────────────
    for p in paths:
        sha256 = hashlib.sha256()
        data   = p.read_bytes()
        sha256.update(data)
        checksums[p.name] = sha256.hexdigest()
        sizes[p.name]     = len(data)

    # ── Write ZIP ─────────────────────────────────────────────────────────────
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=p.name)

    # ── Locate primary artifacts in the supplied list for manifest fields ─────
    cog_path        = _find_artifact(paths, suffix=".tif",  hint="cog")
    stac_item_path  = _find_artifact(paths, suffix=".json", hint="stac")
    audit_log_path  = _find_artifact(paths, suffix=".json", hint="audit")

    return ExportManifest(
        run_id         = run_id,
        exported_at    = datetime.now(timezone.utc).isoformat(),
        cog_path       = cog_path,
        stac_item_path = stac_item_path,
        audit_log_path = audit_log_path,
        zip_path       = zip_path,
        file_checksums = checksums,
        file_sizes     = sizes,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _find_artifact(
    paths:  list[Path],
    suffix: str,
    hint:   str,
) -> Path:
    """
    Return the first path whose suffix matches and whose stem contains *hint*,
    falling back to the first path with the matching suffix, then to Path(".").
    """
    matches = [p for p in paths if p.suffix == suffix]
    hinted  = [p for p in matches if hint in p.stem.lower()]
    if hinted:
        return hinted[0]
    if matches:
        return matches[0]
    return Path(".")
