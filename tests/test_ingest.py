"""
Tests for pipeline/ingest.py (Phase 2A).

Covers
------
- Sensor detection heuristics (filename, band count, tags)
- Acquisition date extraction (tags and filename patterns)
- ingest_path(): happy path TIF, ZIP extraction, error cases
- ingest_upload(): save-then-ingest flow (upload -> disk -> ingest_path)
- Local path validation (existence, suffix, readability)
- Large-file safety: no pixel reads during ingestion
- build_metadata_table(): output shape and keys
"""
from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pipeline.ingest import (
    build_metadata_table,
    detect_sensor,
    extract_acquisition_date,
    ingest_path,
    ingest_upload,
)
from tests.conftest import make_layer, wrap_in_zip


# ── detect_sensor ─────────────────────────────────────────────────────────────

class TestDetectSensor:
    def test_sentinel1_by_filename(self):
        assert detect_sensor("S1A_IW_GRDH_1SDV_20200315.tif", 2, {}) == "Sentinel-1"

    def test_sentinel2_by_filename(self):
        assert detect_sensor("S2B_MSIL2A_20200315T101559.tif", 10, {}) == "Sentinel-2"

    def test_landsat8_by_filename(self):
        result = detect_sensor("LC08_L2SP_193028_20200315.tif", 6, {})
        assert result == "Landsat 8/9"

    def test_landsat9_by_filename(self):
        result = detect_sensor("LC09_L2SP_193028_20210315.tif", 6, {})
        assert result == "Landsat 8/9"

    def test_dem_by_filename(self):
        result = detect_sensor("srtm_dem_30m.tif", 1, {})
        assert result == "DEM"

    def test_dem_by_tags(self):
        result = detect_sensor("elevation.tif", 1, {"description": "digital elevation model"})
        assert result == "DEM"

    def test_single_band_unknown(self):
        result = detect_sensor("mystery_layer.tif", 1, {})
        assert "single-band" in result.lower() or result == "Single-band (SAR or DEM)"

    def test_rgb_unknown(self):
        result = detect_sensor("unnamed.tif", 3, {})
        assert "multispectral" in result.lower() or "rgb" in result.lower()

    def test_hyperspectral_unknown(self):
        result = detect_sensor("unnamed.tif", 12, {})
        assert "multispectral" in result.lower() or "hyper" in result.lower()


# ── extract_acquisition_date ──────────────────────────────────────────────────

class TestExtractAcquisitionDate:
    def test_yyyymmdd_in_filename(self):
        d = extract_acquisition_date("S2A_MSIL2A_20200315T000000.tif", {})
        assert d == date(2020, 3, 15)

    def test_yyyy_mm_dd_in_filename(self):
        d = extract_acquisition_date("scene_2021-07-04.tif", {})
        assert d == date(2021, 7, 4)

    def test_tag_date_acquired(self):
        d = extract_acquisition_date("no_date.tif", {"DATE_ACQUIRED": "2019-11-22"})
        assert d == date(2019, 11, 22)

    def test_tag_sensing_time_iso(self):
        d = extract_acquisition_date(
            "no_date.tif",
            {"SENSING_TIME": "2022-06-01T10:30:00.000Z"},
        )
        assert d == date(2022, 6, 1)

    def test_tag_acquisition_date(self):
        d = extract_acquisition_date("no_date.tif", {"ACQUISITION_DATE": "20230815"})
        assert d == date(2023, 8, 15)

    def test_no_date_returns_none(self):
        d = extract_acquisition_date("nodatefile.tif", {})
        assert d is None

    def test_invalid_date_ignored(self):
        # "19991399" is an invalid date (month 13) — must return None or skip gracefully
        d = extract_acquisition_date("file_19991399.tif", {})
        assert d is None


# ── ingest_path ───────────────────────────────────────────────────────────────

class TestIngestPath:
    """Core path-based ingestion — the primary entry point."""

    def test_ingest_tif_returns_layer_dict(self, make_raster):
        tif_path = make_raster("S2A_20200315_MSIL2A.tif", bands=4, crs_epsg=32633)

        layer = ingest_path(tif_path)

        assert layer["filename"] == tif_path.name
        assert layer["path"] == tif_path
        assert layer["meta"]["count"] == 4
        assert layer["meta"]["crs_epsg"] == 32633
        assert layer["date"] == date(2020, 3, 15)
        assert layer["sensor"] == "Sentinel-2"

    def test_ingest_zip_extracts_tif(self, make_raster, tmp_path):
        tif_path = make_raster("S1A_20210601_GRD.tif", bands=2, crs_epsg=32633)
        zip_path = tmp_path / "S1A_20210601_GRD.zip"
        wrap_in_zip(tif_path, zip_path)

        layer = ingest_path(zip_path, extract_dir=tmp_path)

        assert layer["filename"].endswith(".tif")
        assert layer["meta"]["count"] == 2

    def test_ingest_zip_default_extract_dir(self, make_raster, tmp_path):
        """ZIP without explicit extract_dir extracts into zip's parent."""
        tif_path = make_raster("scene.tif", bands=1)
        zip_path = tmp_path / "scene.zip"
        wrap_in_zip(tif_path, zip_path)

        layer = ingest_path(zip_path)

        assert layer["filename"] == "scene.tif"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="File not found"):
            ingest_path(tmp_path / "nonexistent.tif")

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "image.png"
        bad.write_bytes(b"\x89PNG")
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_path(bad)

    def test_empty_zip_raises(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        with pytest.raises(ValueError, match="No .tif files"):
            ingest_path(zip_path)

    def test_string_path_accepted(self, make_raster):
        """ingest_path must accept a str as well as a Path."""
        tif_path = make_raster("test.tif", bands=1)
        layer = ingest_path(str(tif_path))
        assert layer["filename"] == "test.tif"

    def test_layer_type_set_for_sar(self, make_raster):
        tif_path = make_raster("S1A_IW_20220101.tif", bands=2, crs_epsg=32633)
        layer = ingest_path(tif_path)
        assert layer["layer_type"] == "sar"

    def test_layer_type_set_for_dem(self, make_raster):
        tif_path = make_raster("srtm_dem_30m.tif", bands=1, crs_epsg=4326)
        layer = ingest_path(tif_path)
        assert layer["layer_type"] == "dem"

    def test_no_crs_does_not_raise(self, make_raster):
        tif_path = make_raster("nocrs.tif", crs_epsg=None, bands=1)
        layer = ingest_path(tif_path)
        assert layer["meta"]["crs"] is None
        assert layer["meta"]["crs_epsg"] is None


# ── ingest_upload ─────────────────────────────────────────────────────────────

class TestIngestUpload:
    """Upload → save → ingest_path wrapper."""

    def _make_fake_upload(self, path: Path):
        """Minimal UploadedFile-like object backed by an on-disk file."""
        class FakeUpload:
            name = path.name
            def __init__(self, p):
                self._fh = open(p, "rb")
            def seek(self, n):
                self._fh.seek(n)
            def read(self, n=-1):
                return self._fh.read(n)
        return FakeUpload(path)

    def test_upload_tif_saves_and_ingests(self, make_raster, tmp_path):
        tif_path = make_raster("S2A_20200315_MSIL2A.tif", bands=4, crs_epsg=32633)
        upload   = self._make_fake_upload(tif_path)
        dest_dir = tmp_path / "run_test"

        layer = ingest_upload(upload, dest_dir)

        # File must be persisted to dest_dir
        assert (dest_dir / tif_path.name).exists()
        assert layer["filename"] == tif_path.name
        assert layer["meta"]["count"] == 4
        assert layer["date"] == date(2020, 3, 15)

    def test_upload_zip_extracts_tif(self, make_raster, tmp_path):
        tif_path = make_raster("S1A_20210601_GRD.tif", bands=2, crs_epsg=32633)
        zip_path = tmp_path / "S1A_20210601_GRD.zip"
        wrap_in_zip(tif_path, zip_path)
        upload   = self._make_fake_upload(zip_path)
        dest_dir = tmp_path / "run_zip"

        layer = ingest_upload(upload, dest_dir)

        assert layer["filename"].endswith(".tif")
        assert layer["meta"]["count"] == 2

    def test_upload_empty_zip_raises(self, tmp_path):
        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w"):
            pass
        upload = self._make_fake_upload(zip_path)
        with pytest.raises(ValueError, match="No .tif files"):
            ingest_upload(upload, tmp_path / "run_empty")

    def test_upload_creates_dest_dir(self, make_raster, tmp_path):
        tif_path = make_raster("scene.tif", bands=1)
        upload   = self._make_fake_upload(tif_path)
        dest_dir = tmp_path / "new" / "nested" / "dir"

        assert not dest_dir.exists()
        ingest_upload(upload, dest_dir)
        assert dest_dir.exists()


# ── Local path validation ─────────────────────────────────────────────────────

class TestLocalPathValidation:
    """
    Validate the checks that ingest_path() applies before touching rasterio.
    These mirror the server-side validation that the UI also performs before
    calling ingest_path().
    """

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_path(tmp_path / "does_not_exist.tif")

    def test_directory_path_raises_value_error(self, tmp_path):
        subdir = tmp_path / "adir"
        subdir.mkdir()
        with pytest.raises(ValueError, match="Not a file"):
            ingest_path(subdir)

    def test_wrong_suffix_raises_value_error(self, tmp_path):
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff")   # JPEG magic bytes
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_path(img)

    def test_tif_suffix_accepted(self, make_raster):
        p = make_raster("file.tif", bands=1)
        layer = ingest_path(p)
        assert layer["path"].suffix == ".tif"

    def test_tiff_suffix_accepted(self, make_raster, tmp_path):
        # make_raster always produces .tif; rename to .tiff for this test
        src = make_raster("file.tif", bands=1)
        dst = tmp_path / "file.tiff"
        src.rename(dst)
        layer = ingest_path(dst)
        assert layer["path"].suffix == ".tiff"

    def test_zip_suffix_accepted(self, make_raster, tmp_path):
        tif_path = make_raster("inner.tif", bands=1)
        zip_path = tmp_path / "bundle.zip"
        wrap_in_zip(tif_path, zip_path)
        layer = ingest_path(zip_path)
        assert layer["filename"] == "inner.tif"


# ── Large-file safety ─────────────────────────────────────────────────────────

class TestLargeFilePathSafety:
    """
    Ingestion must remain metadata-only — no pixel reads.

    We verify this by monkey-patching rasterio.open so that any call to
    DatasetReader.read() raises an AssertionError.  If ingest_path() triggers
    a pixel read the test will fail; otherwise it passes cleanly.
    """

    def test_ingest_path_makes_no_pixel_reads(self, make_raster, monkeypatch):
        tif_path = make_raster("large_ortho.tif", bands=3, width=512, height=512)

        import rasterio

        original_open = rasterio.open

        class NoReadDataset:
            """Wraps a real DatasetReader but raises on any .read() call."""
            def __init__(self, path, *args, **kwargs):
                self._ds = original_open(path, *args, **kwargs)

            def read(self, *args, **kwargs):
                raise AssertionError(
                    "ingest_path() must not call .read() — metadata only!"
                )

            def __getattr__(self, name):
                return getattr(self._ds, name)

            def __enter__(self):
                self._ds.__enter__()
                return self

            def __exit__(self, *exc):
                return self._ds.__exit__(*exc)

        monkeypatch.setattr(rasterio, "open", NoReadDataset)
        # If any pixel read occurs, AssertionError propagates and the test fails
        layer = ingest_path(tif_path)

        assert layer["meta"]["width"] == 512
        assert layer["meta"]["height"] == 512

    def test_ingest_upload_makes_no_pixel_reads(self, make_raster, tmp_path, monkeypatch):
        """The upload wrapper must also be pixel-read-free."""
        tif_path = make_raster("S1A_20230101.tif", bands=2)

        class FakeUpload:
            name = tif_path.name
            def __init__(self, p): self._fh = open(p, "rb")
            def seek(self, n): self._fh.seek(n)
            def read(self, n=-1): return self._fh.read(n)

        import rasterio
        original_open = rasterio.open

        class NoReadDataset:
            def __init__(self, path, *args, **kwargs):
                self._ds = original_open(path, *args, **kwargs)
            def read(self, *args, **kwargs):
                raise AssertionError("No pixel reads during ingestion!")
            def __getattr__(self, name):
                return getattr(self._ds, name)
            def __enter__(self):
                self._ds.__enter__()
                return self
            def __exit__(self, *exc):
                return self._ds.__exit__(*exc)

        monkeypatch.setattr(rasterio, "open", NoReadDataset)

        layer = ingest_upload(FakeUpload(tif_path), tmp_path / "run_upload")
        assert layer["meta"]["count"] == 2


# ── build_metadata_table ──────────────────────────────────────────────────────

class TestBuildMetadataTable:
    def test_returns_one_row_per_layer(self, make_raster, tmp_path):
        p1 = make_raster("a.tif", bands=2)
        p2 = make_raster("b.tif", bands=4)
        layers = {
            "sar__a.tif":     make_layer(p1, "sar"),
            "optical__b.tif": make_layer(p2, "optical"),
        }
        rows = build_metadata_table(layers)
        assert len(rows) == 2

    def test_required_keys_present(self, make_raster):
        p = make_raster("x.tif", bands=1)
        layers = {"sar__x.tif": make_layer(p, "sar")}
        row = build_metadata_table(layers)[0]
        for key in ("Layer key", "Filename", "Sensor", "Type", "CRS", "Bands"):
            assert key in row, f"Missing column: {key}"

    def test_no_crs_shows_label(self, make_raster):
        p = make_raster("nocrs.tif", crs_epsg=None)
        layers = {"sar__nocrs.tif": make_layer(p, "sar")}
        row = build_metadata_table(layers)[0]
        assert "No CRS" in row["CRS"] or row["CRS"] == "No CRS"
