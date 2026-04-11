"""
Phase 2A — Light ingestion.

Saves files to disk, extracts lightweight metadata (no pixel reads),
detects sensor type, and parses acquisition dates.

Architecture
------------
The ingestion core is **path-based**.  All processing starts from a file
that already exists on disk:

    ingest_path(path)          ← primary entry point
    ingest_upload(file, dir)   ← convenience wrapper: save → ingest_path

Never pass a Streamlit UploadedFile directly into the pipeline core.

A "layer" is the central data unit throughout the pipeline:
{
    "path":       Path,          # absolute path to the file on disk
    "filename":   str,
    "sensor":     str,           # detected sensor name
    "layer_type": str,           # "sar" | "optical" | "dem" | "unknown"
    "date":       date | None,   # parsed acquisition date
    "meta":       dict,          # from raster_io.get_meta() — zero pixel reads
}
"""
from __future__ import annotations

import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from pipeline.raster_io import get_meta

# ── Sensor detection ──────────────────────────────────────────────────────────

# (compiled regex, sensor label) — evaluated in order; first match wins
_SENSOR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"S1[AB]_",               re.I), "Sentinel-1"),
    (re.compile(r"S2[AB]_",               re.I), "Sentinel-2"),
    (re.compile(r"LC0[89]_|LC08_|LC09_",  re.I), "Landsat 8/9"),
    (re.compile(r"LC07_",                 re.I), "Landsat 7"),
    (re.compile(r"dem|dtm|dsm",              re.I), "DEM"),
]


def detect_sensor(filename: str, band_count: int, tags: dict) -> str:
    """
    Infer sensor type from filename, then band count, then TIFF tags.
    Returns a human-readable sensor name string.
    """
    for pattern, name in _SENSOR_PATTERNS:
        if pattern.search(filename):
            return name

    # Tag-based hints
    tag_str = " ".join(str(v) for v in tags.values()).lower()
    if any(k in tag_str for k in ("elevation", "height", "dtm", "dsm", "dem")):
        return "DEM"
    if "sentinel" in tag_str:
        return "Sentinel (unknown)"
    if "landsat" in tag_str:
        return "Landsat (unknown)"

    # Band-count heuristics as last resort
    if band_count == 1:
        return "Single-band (SAR or DEM)"
    if band_count in (3, 4):
        return "Multispectral RGB/RGBN"
    if band_count >= 5:
        return "Multispectral / Hyperspectral"
    return "Unknown"


def _classify_layer_type(sensor: str, band_count: int) -> str:
    s = sensor.lower()
    if "sentinel-1" in s or "sar" in s:
        return "sar"
    if any(x in s for x in ("dem", "dtm", "dsm", "elevation")):
        return "dem"
    if any(x in s for x in ("sentinel-2", "landsat", "multispectral", "hyperspectral")):
        return "optical"
    # Fallback on band count
    if band_count == 1:
        return "sar"   # most likely SAR or DEM — caller can override
    return "unknown"


# ── Date extraction ───────────────────────────────────────────────────────────

_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"(\d{4})(\d{2})(\d{2})"),       # YYYYMMDD
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),     # YYYY-MM-DD
]

_TAG_DATE_KEYS: list[str] = [
    "ACQUISITION_DATE", "DATE_ACQUIRED", "SENSING_TIME",
    "START_TIME", "PRODUCT_START_TIME", "TIFFTAG_DATETIME",
]

_TAG_DATE_FMTS: list[str] = [
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y%m%d",
]


def extract_acquisition_date(filename: str, tags: dict) -> Optional[date]:
    """
    Parse acquisition date from TIFF tags (preferred) or filename patterns.
    Returns None if no date can be reliably extracted.
    """
    # Tags first — most reliable
    for key in _TAG_DATE_KEYS:
        val = tags.get(key, "")
        if not val:
            continue
        val = str(val).strip()
        for fmt in _TAG_DATE_FMTS:
            try:
                return datetime.strptime(val, fmt).date()
            except ValueError:
                continue

    # Filename patterns
    for pattern in _DATE_PATTERNS:
        m = pattern.search(filename)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError:
                continue

    return None


# ── ZIP handling ──────────────────────────────────────────────────────────────

def extract_tif_from_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """
    Extract all .tif / .tiff files from a ZIP archive into dest_dir.
    Returns list of extracted paths; raises ValueError if none found.
    """
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith((".tif", ".tiff")):
                target = dest_dir / Path(member).name
                target.write_bytes(zf.read(member))
                extracted.append(target)

    if not extracted:
        raise ValueError(f"No .tif files found inside {zip_path.name}")

    return extracted


# ── Core ingestion ────────────────────────────────────────────────────────────

def ingest_path(path: str | Path, extract_dir: Path | None = None) -> dict:
    """
    Ingest a raster file that already exists on disk.

    This is the **primary entry point** for all ingestion.  The file must be
    present on disk before this function is called — no I/O upload happens here.

    Parameters
    ----------
    path        : Absolute (or relative) path to a .tif, .tiff, or .zip file.
    extract_dir : Directory into which ZIP contents are extracted.  Defaults to
                  the parent directory of *path*.

    Returns
    -------
    Layer dict with keys: path, filename, sensor, layer_type, date, meta.

    Raises
    ------
    FileNotFoundError              — path does not exist.
    ValueError                     — unsupported extension, or empty ZIP.
    rasterio.errors.RasterioIOError — file is not a valid raster.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    # Unwrap ZIP → use first contained TIF
    if path.suffix.lower() == ".zip":
        dest = extract_dir if extract_dir is not None else path.parent
        tifs = extract_tif_from_zip(path, dest)
        path = tifs[0]

    if path.suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"Unsupported file type: {path.suffix}")

    meta   = get_meta(path)
    sensor = detect_sensor(path.name, meta["count"], meta.get("tags", {}))
    acq_dt = extract_acquisition_date(path.name, meta.get("tags", {}))
    ltype  = _classify_layer_type(sensor, meta["count"])

    return {
        "path":       path,
        "filename":   path.name,
        "sensor":     sensor,
        "layer_type": ltype,
        "date":       acq_dt,
        "meta":       meta,
    }


def ingest_upload(uploaded_file, tmp_dir: Path) -> dict:
    """
    Convenience wrapper: persist an UploadedFile-like object to *tmp_dir*,
    then delegate to ``ingest_path()``.

    This is the correct pattern for Streamlit browser uploads.  The file is
    written to disk in chunks before any pipeline function is called, so no
    full raster copy lingers in the Streamlit process memory.

    Parameters
    ----------
    uploaded_file : Any file-like object that supports ``.name``, ``.seek()``,
                    and ``.read(n)``.  Streamlit's ``UploadedFile`` qualifies.
    tmp_dir       : Staging directory; created automatically if absent.

    Returns
    -------
    Layer dict — same structure as ``ingest_path()``.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    dest = tmp_dir / uploaded_file.name
    uploaded_file.seek(0)
    with open(dest, "wb") as fh:
        while True:
            chunk = uploaded_file.read(1 << 17)  # 128 KiB chunks
            if not chunk:
                break
            fh.write(chunk)

    return ingest_path(dest, extract_dir=tmp_dir)


# ── Metadata table builder ────────────────────────────────────────────────────

def build_metadata_table(layers: dict[str, dict]) -> list[dict]:
    """
    Flatten layer dicts into a list of row dicts suitable for st.dataframe().
    """
    rows = []
    for key, layer in layers.items():
        m = layer["meta"]
        crs_str = (
            f"EPSG:{m['crs_epsg']}"
            if m.get("crs_epsg")
            else (str(m["crs"]) if m.get("crs") else "No CRS")
        )
        rows.append({
            "Layer key":      key,
            "Filename":       layer["filename"],
            "Sensor":         layer["sensor"],
            "Type":           layer["layer_type"],
            "Date":           str(layer["date"]) if layer["date"] else "Unknown",
            "CRS":            crs_str,
            "Bands":          m["count"],
            "Width (px)":     m["width"],
            "Height (px)":    m["height"],
            "GSD (CRS units)": f"{abs(m['res'][0]):.4f}",
            "NoData":         str(m["nodata"]),
        })
    return rows
