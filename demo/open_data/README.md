# TerraForge Demo — Open Data Download Guide

All datasets used in this demo are freely available under open licences.
Follow the instructions below to reproduce the full pipeline run.

---

## Dataset A — Pit Wall Classification (Mount Tom Price, WA)

**Site:** Mount Tom Price Iron Ore Mine, Western Australia  
**Coordinates:** Lat −22.6931 / Lon 117.7939  
**Sentinel-2 tile:** 50JNL  
**Target season:** May–September (dry season, < 5% cloud)

### Required Files

| File | Description | Source |
|------|-------------|--------|
| `S2_B02.tif` | Blue (10 m) | Copernicus Data Space |
| `S2_B03.tif` | Green (10 m) | Copernicus Data Space |
| `S2_B04.tif` | Red (10 m) | Copernicus Data Space |
| `S2_B08.tif` | NIR (10 m) | Copernicus Data Space |
| `S2_B8A.tif` | RedEdge (20 m) | Copernicus Data Space |
| `S2_B11.tif` | SWIR1 (20 m) | Copernicus Data Space |
| `S2_B12.tif` | SWIR2 (20 m) | Copernicus Data Space |
| `DEM_GLO30.tif` | Elevation 30 m | Copernicus DEM / AWS |

---

## Option 1 — Manual Download via Copernicus Browser (No Scripting)

1. Open **https://browser.dataspace.copernicus.eu/**
2. Sign in (free account required).
3. In the search box, enter: `Mount Tom Price`
4. Set filter:
   - **Data collection:** Sentinel-2 L2A
   - **Date range:** 2024-06-01 to 2024-08-31
   - **Max cloud cover:** 5%
5. Select a result with **TILE_ID = 50JNL**.
6. Click **Visualise**, then **Download product**.
7. In the ZIP, locate the `GRANULE/…/IMG_DATA/R10m/` folder.
8. Extract `B02_10m.jp2`, `B03_10m.jp2`, `B04_10m.jp2`, `B08_10m.jp2`.
9. From `R20m/`: extract `B8A_20m.jp2`, `B11_20m.jp2`, `B12_20m.jp2`.
10. Convert `.jp2` → `.tif` with GDAL:

```bash
gdal_translate -of GTiff B02_10m.jp2 demo/open_data/S2_B02.tif
```

Repeat for all 7 bands.  Resample 20 m bands to 10 m if stacking:

```bash
gdalwarp -tr 10 10 -r bilinear B11_20m.jp2 demo/open_data/S2_B11.tif
```

---

## Option 2 — Scripted Download with sentinelsat

```bash
pip install sentinelsat

python3 - << 'EOF'
from sentinelsat import SentinelAPI
from datetime import date

api = SentinelAPI(
    'YOUR_EMAIL',
    'YOUR_PASSWORD',
    'https://apihub.copernicus.eu/apihub'
)

products = api.query(
    area='POINT(117.7939 -22.6931)',
    date=('20240601', '20240831'),
    platformname='Sentinel-2',
    processinglevel='Level-2A',
    cloudcoverpercentage=(0, 5),
)

api.download_all(products, directory_path='demo/open_data/raw/')
EOF
```

---

## Option 3 — pystac-client (Copernicus STAC Endpoint)

```bash
pip install pystac-client odc-stac

python3 - << 'EOF'
from pystac_client import Client

catalog = Client.open("https://catalogue.dataspace.copernicus.eu/stac")

search = catalog.search(
    collections=["SENTINEL-2"],
    bbox=[117.6, -22.8, 117.9, -22.5],
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 5}},
    max_items=3,
)

for item in search.items():
    print(item.id, item.properties.get("eo:cloud_cover"))
EOF
```

---

## Dataset B — TSF Classification (Super Pit, Kalgoorlie-Boulder, WA)

**Site:** Fimiston Open Pit (Super Pit)  
**Coordinates:** Lat −30.7775 / Lon 121.5107  
**Sentinel-2 tile:** 51JZG  

Same download procedure as Dataset A.  
For TSF classification also download **Sentinel-1 GRD** (VV+VH, IW mode):

1. In Copernicus Browser → Data collection: **Sentinel-1**
2. Sensor mode: **IW**, Polarisation: **VV+VH**
3. Same date window (dry season minimises vegetation confusion)

---

## Copernicus DEM GLO-30 (via AWS Open Data)

The GLO-30 DEM is publicly available without credentials:

```bash
# Single tile for Mount Tom Price (tile naming: N22E117)
aws s3 cp \
  s3://copernicus-dem-30m/Copernicus_DSM_COG_10_S23_00_E117_00_DEM/ \
  demo/open_data/ \
  --recursive --no-sign-request

# Or use py-dem-stitcher (recommended — handles tile stitching automatically)
pip install py-dem-stitcher

python3 - << 'EOF'
from dem_stitcher import stitch_dem
import numpy as np

bounds = [117.6, -23.0, 118.0, -22.4]  # [west, south, east, north]
X, p = stitch_dem(bounds, dem_name='glo_30', dst_ellipsoidal_height=False)
import rasterio
with rasterio.open('demo/open_data/DEM_GLO30.tif', 'w', **p) as dst:
    dst.write(X, 1)
EOF
```

---

## Expected Directory Layout After Download

```
demo/open_data/
├── S2_B02.tif      # Blue 10m
├── S2_B03.tif      # Green 10m
├── S2_B04.tif      # Red 10m
├── S2_B08.tif      # NIR 10m
├── S2_B8A.tif      # RedEdge (resampled to 10m)
├── S2_B11.tif      # SWIR1 (resampled to 10m)
├── S2_B12.tif      # SWIR2 (resampled to 10m)
├── DEM_GLO30.tif   # Elevation 30m (resampled to 10m for stacking)
├── S1_VV.tif       # Sentinel-1 VV (optional, TSF only)
└── S1_VH.tif       # Sentinel-1 VH (optional, TSF only)
```

---

## Preprocessing Note

Before running the pipeline, stack all bands into a single multi-band GeoTIFF
**and ensure all bands share the same CRS, extent and resolution**:

```bash
# Stack 7 optical bands into one file (band order matches sentinel2_l2a_20m template)
gdalbuildvrt -separate stack.vrt S2_B02.tif S2_B03.tif S2_B04.tif \
    S2_B8A.tif S2_B11.tif S2_B12.tif
gdal_translate -of GTiff stack.vrt demo/open_data/S2_stack_7band.tif
```

The demo script (`demo/pit_wall_demo.py`) accepts the stacked file directly.

---

## Data Licences

| Dataset | Licence |
|---------|---------|
| Sentinel-2 L2A | Copernicus Open Access — free reuse with attribution |
| Sentinel-1 GRD | Copernicus Open Access — free reuse with attribution |
| Copernicus DEM GLO-30 | Copernicus DEM licence (free for commercial/research use) |

Attribution: *Contains modified Copernicus Sentinel data [year].*
