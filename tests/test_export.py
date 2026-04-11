"""
Phase 7 — Export system tests.

All rasters are created in-process; no external files required.
GDAL-dependent tests (write_cog) are skipped if osgeo is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pipeline.export import (
    ExportManifest,
    _find_artifact,
    build_stac_item,
    export_audit_log,
    package_run,
    write_stac_item,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_tif(
    tmp_path:  Path,
    filename:  str   = "classified.tif",
    dtype:     str   = "int16",
    nodata:    float | None = -1.0,
    crs_epsg:  int   = 32633,
    width:     int   = 20,
    height:    int   = 20,
    res:       float = 10.0,
) -> Path:
    """Write a minimal synthetic GeoTIFF."""
    path      = tmp_path / filename
    transform = from_origin(500_000.0, 5_000_200.0, res, res)
    rng       = np.random.default_rng(0)
    data      = rng.integers(1, 5, size=(1, height, width)).astype(dtype)
    profile: dict = {
        "driver":    "GTiff",
        "dtype":     dtype,
        "width":     width,
        "height":    height,
        "count":     1,
        "transform": transform,
        "crs":       CRS.from_epsg(crs_epsg) if crs_epsg else None,
        "nodata":    nodata,
    }
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── write_cog ─────────────────────────────────────────────────────────────────

class TestWriteCog:

    @pytest.fixture(autouse=True)
    def _require_gdal(self) -> None:
        pytest.importorskip("osgeo.gdal", reason="GDAL not available")

    def test_output_file_created(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "src.tif")
        out = write_cog(src, tmp_path / "cog.tif")
        assert out.exists()

    def test_crs_and_transform_preserved(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "src.tif", crs_epsg=32633, res=10.0)
        out = write_cog(src, tmp_path / "cog.tif")
        with rasterio.open(src) as ds_src, rasterio.open(out) as ds_out:
            assert ds_out.crs == ds_src.crs
            assert ds_out.transform == ds_src.transform

    def test_nodata_preserved_when_present(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "src.tif", dtype="int16", nodata=-1.0)
        out = write_cog(src, tmp_path / "cog.tif")
        with rasterio.open(out) as ds:
            assert ds.nodata == pytest.approx(-1.0)

    def test_nodata_assigned_int_when_missing(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "src.tif", dtype="int16", nodata=None)
        out = write_cog(src, tmp_path / "cog.tif")
        with rasterio.open(out) as ds:
            assert ds.nodata == pytest.approx(-1.0)

    def test_nodata_assigned_float_when_missing(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "f32.tif", dtype="float32", nodata=None)
        out = write_cog(src, tmp_path / "cog_f32.tif")
        with rasterio.open(out) as ds:
            assert ds.nodata == pytest.approx(-9999.0)

    def test_overviews_present(self, tmp_path: Path) -> None:
        from pipeline.export import write_cog
        # Need a raster large enough to build overviews for all 4 levels (≥16 px)
        src = _make_tif(tmp_path, "big.tif", width=64, height=64)
        out = write_cog(src, tmp_path / "cog_big.tif")
        with rasterio.open(out) as ds:
            assert ds.overviews(1), "COG must have at least one overview level"

    def test_float32_uses_predictor_3(self, tmp_path: Path) -> None:
        """GDAL tag PREDICTOR should be 3 for float32 output."""
        from osgeo import gdal
        from pipeline.export import write_cog
        src = _make_tif(tmp_path, "f32.tif", dtype="float32")
        out = write_cog(src, tmp_path / "cog_f32.tif")
        ds  = gdal.Open(str(out))
        md  = ds.GetMetadata("IMAGE_STRUCTURE")
        ds  = None
        assert md.get("PREDICTOR") == "3"


# ── build_stac_item ───────────────────────────────────────────────────────────

class TestBuildStacItem:

    @pytest.fixture(autouse=True)
    def _require_pyproj(self) -> None:
        pytest.importorskip("pyproj", reason="pyproj not available")

    @pytest.fixture
    def cog(self, tmp_path: Path) -> Path:
        return _make_tif(tmp_path, "classified_cog.tif", crs_epsg=32633)

    def test_required_stac_keys_present(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        for key in ("type", "stac_version", "id", "geometry", "bbox",
                    "properties", "links", "assets"):
            assert key in item, f"Missing STAC key: {key}"

    def test_stac_version_is_1_0(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert item["stac_version"] == "1.0.0"

    def test_id_contains_run_id(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert "ABCD1234" in item["id"]

    def test_bbox_is_four_floats(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        bbox = item["bbox"]
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)

    def test_bbox_order_west_south_east_north(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        west, south, east, north = item["bbox"]
        assert west < east
        assert south < north

    def test_bbox_in_wgs84_range(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        west, south, east, north = item["bbox"]
        assert -180.0 <= west <= 180.0
        assert -90.0  <= south <= 90.0
        assert -180.0 <= east  <= 180.0
        assert -90.0  <= north <= 90.0

    def test_geometry_is_closed_polygon(self, cog: Path) -> None:
        item   = build_stac_item(cog, run_id="ABCD1234")
        coords = item["geometry"]["coordinates"][0]
        assert item["geometry"]["type"] == "Polygon"
        assert len(coords) == 5              # 4 corners + closing point
        assert coords[0] == coords[-1]       # closed ring

    def test_geometry_coords_are_native_float(self, cog: Path) -> None:
        item   = build_stac_item(cog, run_id="ABCD1234")
        coords = item["geometry"]["coordinates"][0]
        for lon, lat in coords:
            assert type(lon) is float
            assert type(lat) is float

    def test_run_id_in_properties(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert item["properties"]["run_id"] == "ABCD1234"

    def test_datetime_present(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert "datetime" in item["properties"]

    def test_proj_epsg_present(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert "proj:epsg" in item["properties"]
        assert item["properties"]["proj:epsg"] == 32633

    def test_proj_shape_is_rows_cols(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        shape = item["properties"]["proj:shape"]
        assert isinstance(shape, list) and len(shape) == 2
        assert shape == [20, 20]   # height=20, width=20 from fixture

    def test_extra_properties_merged(self, cog: Path) -> None:
        item = build_stac_item(
            cog, run_id="ABCD1234",
            properties={"classification:classes": [1, 2, 3]}
        )
        assert item["properties"]["classification:classes"] == [1, 2, 3]

    def test_reserved_properties_override_caller(self, cog: Path) -> None:
        """run_id supplied by caller must be overridden by the function's value."""
        item = build_stac_item(
            cog, run_id="ABCD1234",
            properties={"run_id": "SPOOFED", "proj:epsg": 99999}
        )
        assert item["properties"]["run_id"]   == "ABCD1234"
        assert item["properties"]["proj:epsg"] == 32633

    def test_data_asset_present(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        assert "data" in item["assets"]
        asset = item["assets"]["data"]
        assert "href" in asset
        assert "cloud-optimized" in asset["type"]

    def test_projection_extension_declared(self, cog: Path) -> None:
        item = build_stac_item(cog, run_id="ABCD1234")
        exts = item["stac_extensions"]
        assert any("projection" in e for e in exts)


# ── write_stac_item ───────────────────────────────────────────────────────────

class TestWriteStacItem:

    def test_file_created(self, tmp_path: Path) -> None:
        item = {"type": "Feature", "id": "test"}
        out  = write_stac_item(item, tmp_path / "item.json")
        assert out.exists()

    def test_valid_json_roundtrip(self, tmp_path: Path) -> None:
        item = {"type": "Feature", "id": "test", "nested": {"a": 1}}
        out  = write_stac_item(item, tmp_path / "item.json")
        loaded = json.loads(out.read_text())
        assert loaded == item

    def test_pretty_printed(self, tmp_path: Path) -> None:
        item = {"a": 1}
        out  = write_stac_item(item, tmp_path / "item.json")
        assert "\n" in out.read_text()


# ── export_audit_log ──────────────────────────────────────────────────────────

class TestExportAuditLog:

    def test_file_created(self, tmp_path: Path) -> None:
        out = export_audit_log("FAKEID", tmp_path / "audit.json")
        assert out.exists()

    def test_valid_json_list(self, tmp_path: Path) -> None:
        out  = export_audit_log("FAKEID", tmp_path / "audit.json")
        data = json.loads(out.read_text())
        assert isinstance(data, list)

    def test_empty_log_exports_empty_list(self, tmp_path: Path) -> None:
        out  = export_audit_log("NONEXISTENT_RUN_XYZ", tmp_path / "audit.json")
        data = json.loads(out.read_text())
        assert data == []


# ── package_run ───────────────────────────────────────────────────────────────

class TestPackageRun:

    @pytest.fixture
    def artifacts(self, tmp_path: Path) -> list[Path]:
        """Three minimal artifact files."""
        cog   = _make_tif(tmp_path, "classified_cog.tif")
        stac  = tmp_path / "stac_item.json"
        audit = tmp_path / "audit_log.json"
        stac.write_text(json.dumps({"type": "Feature"}))
        audit.write_text(json.dumps([{"event": "run_start"}]))
        return [cog, stac, audit]

    def test_zip_created(self, tmp_path: Path, artifacts: list[Path]) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        assert manifest.zip_path.exists()

    def test_zip_contains_all_artifacts(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        with zipfile.ZipFile(manifest.zip_path) as zf:
            names = zf.namelist()
        for p in artifacts:
            assert p.name in names

    def test_checksums_are_64_char_hex(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        for checksum in manifest.file_checksums.values():
            assert len(checksum) == 64
            assert all(c in "0123456789abcdef" for c in checksum)

    def test_checksums_match_source_files(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        for p in artifacts:
            assert manifest.file_checksums[p.name] == _sha256(p)

    def test_file_sizes_match_source_files(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        for p in artifacts:
            assert manifest.file_sizes[p.name] == p.stat().st_size

    def test_checksums_computed_before_zip(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        """Checksum of the .tif must match the pre-zip source, not the zip entry."""
        tif = artifacts[0]
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        assert manifest.file_checksums[tif.name] == _sha256(tif)

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.tif"
        with pytest.raises(ValueError, match="Artifact not found"):
            package_run("ABCD1234", [missing], tmp_path / "export")

    def test_zip_name_contains_run_id(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        assert "ABCD1234" in manifest.zip_path.name

    def test_manifest_exported_at_is_iso8601(
        self, tmp_path: Path, artifacts: list[Path]
    ) -> None:
        from datetime import datetime
        manifest = package_run("ABCD1234", artifacts, tmp_path / "export")
        # Should parse without raising
        datetime.fromisoformat(manifest.exported_at)

    def test_additional_artifacts_included(self, tmp_path: Path) -> None:
        """Dynamic discovery: extra files beyond the canonical three are included."""
        f1 = tmp_path / "extra_features.tif"
        f2 = tmp_path / "stac_item.json"
        f3 = tmp_path / "audit_log.json"
        f4 = tmp_path / "drift_report.json"
        # Write minimal content to each
        _make_tif(tmp_path, "extra_features.tif")
        f2.write_text(json.dumps({"type": "Feature"}))
        f3.write_text(json.dumps([]))
        f4.write_text(json.dumps({"flagged": []}))
        manifest = package_run("ZZZZ9999", [f1, f2, f3, f4], tmp_path / "export")
        with zipfile.ZipFile(manifest.zip_path) as zf:
            names = zf.namelist()
        assert "extra_features.tif" in names
        assert "drift_report.json" in names


# ── _find_artifact (internal) ─────────────────────────────────────────────────

class TestFindArtifact:

    def test_hint_matched_first(self, tmp_path: Path) -> None:
        a = tmp_path / "stac_item.json"
        b = tmp_path / "audit_log.json"
        a.touch(); b.touch()
        result = _find_artifact([a, b], suffix=".json", hint="stac")
        assert result == a

    def test_falls_back_to_first_with_suffix(self, tmp_path: Path) -> None:
        a = tmp_path / "other.json"
        a.touch()
        result = _find_artifact([a], suffix=".json", hint="stac")
        assert result == a

    def test_returns_dot_when_no_suffix_match(self, tmp_path: Path) -> None:
        a = tmp_path / "file.tif"
        a.touch()
        result = _find_artifact([a], suffix=".json", hint="stac")
        assert result == Path(".")
