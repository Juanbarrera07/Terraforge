"""
demo/labels/create_demo_labels.py
==================================
Generate a geotechnically coherent synthetic label raster for any input extent.

Class distribution (pit wall scenario):
  1 Exposed Fresh Rock      35-45% -- central pit floor / inner wall
  2 Weathered/Oxidised      25-35% -- mid-slope weathering zone
  3 Instability Zone         5-10% -- localised transition patches
  4 Vegetation Encroachment 10-15% -- outer boundary
  5 Water / Seepage          5-10% -- pit floor depression

All spatial structure is generated from layered noise and distance transforms
so the result looks geologically plausible even without real field data.
random_state is fixed at 42 for full reproducibility.

Usage
-----
  python3 demo/labels/create_demo_labels.py --src /path/to/raster.tif \
      --out demo/labels/label.tif [--mode pit_wall|tsf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _rng_seed42():
    import numpy as np
    return np.random.default_rng(42)


def _gaussian_blob(H, W, cy, cx, sigma, rng):
    """Return (H, W) array with Gaussian peak at (cy, cx)."""
    import numpy as np
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _perlin_noise(H, W, scale, rng):
    """Cheap band-limited noise: sum of downsampled random grids."""
    import numpy as np
    from scipy.ndimage import zoom

    out = np.zeros((H, W), dtype=np.float64)
    amplitude = 1.0
    for freq in [1, 2, 4, 8]:
        sh = max(1, int(H / scale * freq))
        sw = max(1, int(W / scale * freq))
        tile = rng.random((sh, sw))
        zoomed = zoom(tile, (H / sh, W / sw), order=1)
        zoomed = zoomed[:H, :W]
        if zoomed.shape != (H, W):
            pad_h = H - zoomed.shape[0]
            pad_w = W - zoomed.shape[1]
            zoomed = np.pad(zoomed, ((0, pad_h), (0, pad_w)), mode="edge")
        out += amplitude * zoomed
        amplitude *= 0.5
    out -= out.min()
    out /= (out.max() + 1e-9)
    return out


def create_pit_wall_labels(H: int, W: int) -> "np.ndarray":
    """
    Generate a (H, W) int16 pit wall label array with realistic spatial structure.

    Class layout:
    - Water (5):       elliptical depression at pit floor (bottom-centre)
    - Rock (1):        inner pit -- high slope / low NDVI proxy zone
    - Instability (3): narrow irregular band along the inner wall edge
    - Weathered (2):   mid-slope weathering mantle
    - Vegetation (4):  outer edge / crest zone
    """
    import numpy as np

    rng = _rng_seed42()

    cy, cx = H * 0.55, W * 0.5
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dist = np.sqrt(((yy - cy) / (H * 0.5)) ** 2 + ((xx - cx) / (W * 0.5)) ** 2)

    noise = _perlin_noise(H, W, scale=8, rng=rng)
    dist_p = dist + 0.15 * (noise - 0.5)

    labels = np.ones((H, W), dtype=np.int16)

    labels[dist_p > 0.78] = 4
    labels[(dist_p > 0.45) & (dist_p <= 0.78)] = 2
    labels[(dist_p > 0.18) & (dist_p <= 0.45)] = 1

    instab_noise = _perlin_noise(H, W, scale=4, rng=rng)
    instab_mask  = (dist_p > 0.42) & (dist_p <= 0.50) & (instab_noise > 0.45)
    labels[instab_mask] = 3

    water_cy   = H * 0.70
    water_blob = _gaussian_blob(H, W, water_cy, cx, sigma=H * 0.07, rng=rng)
    labels[(water_blob > 0.55) & (dist_p < 0.25)] = 5

    return labels


def create_tsf_labels(H: int, W: int) -> "np.ndarray":
    """
    Generate (H, W) int16 TSF label array.

    Classes:
      1 Active Wet Tailings  -- central impoundment
      2 Dry Beach            -- wide perimeter beach
      3 Ponded Water         -- low point / decant pond
      4 Rehabilitated        -- outer crest (capped)
      5 Dust Risk Zone       -- downwind dry face
      6 Seepage Zone         -- toe seepage near embankment
    """
    import numpy as np

    rng = _rng_seed42()

    cy, cx = H * 0.45, W * 0.5
    y = np.arange(H, dtype=np.float64)
    x = np.arange(W, dtype=np.float64)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dist = np.sqrt(((yy - cy) / (H * 0.48)) ** 2 + ((xx - cx) / (W * 0.52)) ** 2)

    noise  = _perlin_noise(H, W, scale=8, rng=rng)
    dist_p = dist + 0.12 * (noise - 0.5)

    labels = np.full((H, W), 4, dtype=np.int16)

    labels[(dist_p > 0.30) & (dist_p <= 0.72)] = 2
    labels[dist_p <= 0.30] = 1
    labels[(dist_p <= 0.12) & (yy > H * 0.55)] = 3

    seep_noise = _perlin_noise(H, W, scale=6, rng=rng)
    seep_mask  = (dist_p > 0.68) & (dist_p <= 0.80) & (seep_noise > 0.55)
    labels[seep_mask] = 6

    dust_mask  = (dist_p > 0.22) & (dist_p <= 0.50) & (xx > cx + W * 0.05)
    dust_noise = _perlin_noise(H, W, scale=5, rng=rng)
    labels[dust_mask & (dust_noise > 0.60)] = 5

    return labels


def write_label_raster(labels: "np.ndarray", src_path: Path, out_path: Path) -> None:
    """Write labels as a GeoTIFF with the same CRS/transform as src_path."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        src_h, src_w = src.height, src.width

    H, W = labels.shape
    profile.update({
        "count":    1,
        "dtype":    "int16",
        "nodata":   0,
        "compress": "deflate",
        "predictor": 2,
        "height": H,
        "width":  W,
    })
    if src_h != H or src_w != W:
        with rasterio.open(src_path) as src:
            bounds = src.bounds
        profile["transform"] = from_bounds(
            bounds.left, bounds.bottom, bounds.right, bounds.top, W, H
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(labels.astype(np.int16), 1)


def print_class_distribution(labels: "np.ndarray", mode: str) -> None:
    import numpy as np
    CLASS_NAMES = {
        "pit_wall": {
            1: "Exposed Fresh Rock",
            2: "Weathered/Oxidised Material",
            3: "Instability Zone",
            4: "Vegetation Encroachment",
            5: "Water / Seepage",
        },
        "tsf": {
            1: "Active Wet Tailings",
            2: "Dry Beach (Oxidised)",
            3: "Ponded Water",
            4: "Rehabilitated Surface",
            5: "Dust Risk Zone",
            6: "Seepage / Discolouration",
        },
    }
    names = CLASS_NAMES.get(mode, {})
    total = labels.size
    print(f"  Class distribution ({mode} mode):")
    print(f"  {'ID':<4} {'Name':<35} {'Pixels':>8} {'%':>6}")
    print(f"  " + "-" * 56)
    for cls_id in sorted(np.unique(labels)):
        if cls_id == 0:
            continue
        count = int(np.sum(labels == cls_id))
        pct   = 100.0 * count / total
        name  = names.get(int(cls_id), f"Class {cls_id}")
        print(f"  {cls_id:<4} {name:<35} {count:>8,} {pct:>5.1f}%")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate synthetic but geotechnically coherent label rasters."
    )
    ap.add_argument("--src",  required=True,  help="Source raster (extent/CRS reference)")
    ap.add_argument("--out",  required=True,  help="Output label GeoTIFF path")
    ap.add_argument("--mode", default="pit_wall", choices=["pit_wall", "tsf"],
                    help="Classification scheme (default: pit_wall)")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)

    if not src_path.exists():
        print(f"ERROR: source raster not found: {src_path}", file=sys.stderr)
        return 1

    try:
        import rasterio
        with rasterio.open(src_path) as ds:
            H, W = ds.height, ds.width
    except Exception as exc:
        print(f"ERROR: cannot read source raster: {exc}", file=sys.stderr)
        return 1

    print(f"  Generating {args.mode} labels for {W} x {H} raster...")

    if args.mode == "pit_wall":
        labels = create_pit_wall_labels(H, W)
    else:
        labels = create_tsf_labels(H, W)

    print_class_distribution(labels, args.mode)
    write_label_raster(labels, src_path, out_path)
    print(f"  Label raster written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
