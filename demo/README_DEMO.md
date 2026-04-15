# TerraForge Mining Intelligence — Technical Demo

**Automated geospatial material classification for open-pit mining and tailings management.**

This demo applies a production-grade ML classification pipeline to characterise surface materials
in open pit mining environments. The pipeline processes multispectral satellite imagery
(Sentinel-2 L2A) and/or UAV RGB imagery to produce classified maps with quantified material
distribution, confidence assessment, and a full audit trail.

---

## 1. Technical Overview

TerraForge Mining Intelligence automates the most error-prone steps in a geospatial consulting
workflow: band stacking, atmospheric correction, spectral feature computation, model training,
class-area statistics, and PDF report generation. All operations are windowed — no full raster
arrays are held in memory — making the pipeline viable on standard workstation hardware for
imagery up to survey-scale extents.

**Pipeline stages:**

```
Ingest → Validate → Preprocess → Features → Train → Predict → Post-process → Export
```

Each stage writes outputs to disk and appends a timestamped entry to a JSON audit log.
Quality gates at the classification stage block downstream processing if OA or minority F1
fall below configured thresholds.

---

## 2. Classification System

### Pit Wall Classification (Sentinel-2 / UAV)

| ID | Class Name | Definition | Geotechnical Significance |
|----|-----------|------------|--------------------------|
| 1 | Exposed Fresh Rock | Low NDVI, high reflectance, low clay index | Stable face material; baseline condition |
| 2 | Weathered / Oxidised Material | Elevated iron oxide ratio, reddish hue | Reduced shear strength; monitor for progression |
| 3 | Instability Zone | Anomalous texture, high roughness proxy, spectral displacement signature | Potential slope failure risk — field inspection within 48 h |
| 4 | Vegetation Encroachment | High NDVI / ExG, green reflectance | Indicates uncontrolled revegetation; drainage risk |
| 5 | Water / Seepage | High NDWI / mNDWI, low SWIR reflectance | Active seepage point; immediate geotechnical review |
| 0 | NoData / Shadow | Shadow, cloud, or missing data | Excluded from area statistics |

### Tailings Storage Facility Classification (Sentinel-2 + optional SAR)

| ID | Class Name | Definition | Regulatory Significance |
|----|-----------|------------|------------------------|
| 1 | Active Wet Tailings | High moisture index, low backscatter (SAR) | MAC TSF guidelines: freeboard monitoring zone |
| 2 | Dry Beach (Oxidised) | Elevated BSI, high SWIR, low moisture | Dust risk source; DMIRS reporting if > threshold area |
| 3 | Ponded Water | Very high NDWI, smooth SAR texture | Decant management; ANCOLD freeboard compliance |
| 4 | Rehabilitated Surface | High NDVI, stable spectral signature over time | Closure criterion tracking |
| 5 | Dust Risk Zone | Dry, fine-grained, low cohesion spectral signal | Dust suppression trigger zone |
| 6 | Seepage / Discolouration | Anomalous colour index at embankment toe | DMIRS mandatory reporting trigger |
| 0 | NoData | Shadow, cloud, or missing data | Excluded |

---

## 3. Spectral Feature Engineering

Features are computed tile-by-tile (windowed I/O) to avoid full raster loads.

| Feature | Formula | Key Use |
|---------|---------|---------|
| NDVI | (NIR − Red) / (NIR + Red) | Vegetation detection |
| NDWI | (Green − NIR) / (Green + NIR) | Open water |
| BSI | ((SWIR + Red) − (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue)) | Bare soil / dry tailings |
| NDRE | (RedEdge − Red) / (RedEdge + Red) | Early vegetation stress |
| Iron Oxide Ratio | Red / Blue | Weathered / oxidised rock |
| Clay Index | SWIR / NIR | Hydrothermal alteration |
| Ferrous Index | NIR / SWIR | Fresh vs. weathered rock discrimination |
| mNDWI | (Green − SWIR) / (Green + SWIR) | Turbid / sediment-laden water |
| Roughness Proxy | Std(slope) in 3×3 window | Displacement zones on pit wall |
| ExG (drone) | 2·Green − Red − Blue | RGB vegetation proxy |
| ExR (drone) | 1.4·Red − Green | RGB oxidation proxy |
| Brightness (drone) | (R + G + B) / 3 | Reflectance surrogate |
| RGB Hue / Saturation | HSV from numpy | Material colour discrimination |

**Note:** For drone imagery (no NIR/SWIR), a dedicated RGB feature set is used. Bands are
automatically scaled from uint8 to float32 before index computation.

---

## 4. Accuracy and Validation

Validation is performed using stratified k-fold cross-validation (k=5 by default) with
out-of-fold predictions. No data leakage: SMOTE resampling is applied inside each fold,
not before the train/test split.

**Quality gate thresholds (default):**

| Metric | Threshold |
|--------|-----------|
| Overall Accuracy (OA) | ≥ 0.80 |
| Minority class F1 | ≥ 0.70 |
| Coregistration RMSE | ≤ 0.5 px |
| Drift alert | > 20% area change vs. reference |

**Known limitations:**

Spectral confusion between fresh rock and dry tailings beach is expected in single-date
imagery without SWIR bands. SAR backscatter (Sentinel-1 VV/VH) significantly improves
wet/dry discrimination for TSF classification. Instability Zone detection with satellite
data is indicative only — field verification is required before geotechnical decisions.

---

## 5. Outputs

| File | Format | Description | How to Open |
|------|--------|-------------|-------------|
| `{run_id}_classified_cog.tif` | Cloud Optimized GeoTIFF (int16) | Per-pixel integer class map | QGIS, ArcGIS Pro, Google Earth Engine |
| `{run_id}_confidence_cog.tif` | COG float32 | Per-pixel max-class probability 0–1 | QGIS with RdYlGn colour ramp |
| `{run_id}_stac_item.json` | STAC 1.0 | Spatial metadata catalog entry | STAC Browser, intake-stac, Python |
| `{run_id}_report.pdf` | PDF | 6-section technical classification report | Any PDF reader |
| `{run_id}_audit_log.json` | JSON | Complete processing trace with timestamps | Text editor, `python -m json.tool` |
| `{run_id}_export.zip` | ZIP | All artifacts with SHA-256 checksums | Any archive manager |

---

## 6. How to Reproduce

### System Requirements

- Python 3.10+
- GDAL ≥ 3.4 (required for COG export and sieve filter)
- 8 GB RAM minimum (16 GB recommended for full Sentinel-2 tiles)
- ~2 GB disk space for a single Sentinel-2 tile run

### Installation

```bash
# Clone or navigate to repo root
cd /path/to/TerraForge

# Conda environment (recommended)
conda env create -f requirements-conda.txt
conda activate terraforge

# Or pip
pip install -r requirements.txt
```

### Download Open Data

Follow the instructions in `demo/open_data/README.md` to download:
- Sentinel-2 L2A bands for Mount Tom Price (tile 50JNL, May–Sep dry season)
- Copernicus DEM GLO-30 for the site extent

Then stack the 7 optical bands into a single GeoTIFF:

```bash
gdalbuildvrt -separate demo/open_data/stack.vrt \
    demo/open_data/S2_B02.tif demo/open_data/S2_B03.tif \
    demo/open_data/S2_B04.tif demo/open_data/S2_B8A.tif \
    demo/open_data/S2_B11.tif demo/open_data/S2_B12.tif

gdal_translate -of GTiff demo/open_data/stack.vrt \
    demo/open_data/S2_stack_7band.tif
```

### Run the Demo

**Satellite — Pit Wall:**
```bash
python demo/pit_wall_demo.py \
    --site tom_price \
    --mode pit_wall \
    --data-dir demo/open_data/ \
    --output-dir demo/outputs/
```

**Satellite — TSF:**
```bash
python demo/pit_wall_demo.py \
    --site super_pit \
    --mode tsf \
    --data-dir demo/open_data/ \
    --output-dir demo/outputs/
```

**Drone RGB (DJI Air 3S or similar):**
```bash
python demo/pit_wall_demo.py \
    --site custom \
    --mode pit_wall \
    --data-dir /path/to/drone/orthomosaic/ \
    --output-dir demo/outputs/ \
    --drone
```

**Custom label raster:**
```bash
python demo/pit_wall_demo.py \
    --site tom_price \
    --mode pit_wall \
    --data-dir demo/open_data/ \
    --output-dir demo/outputs/ \
    --labels /path/to/field_labels.tif
```

### Generate Synthetic Labels Only

```bash
python demo/labels/create_demo_labels.py \
    --src demo/open_data/S2_stack_7band.tif \
    --out demo/labels/pit_wall_labels.tif \
    --mode pit_wall
```

### Expected Run Times

| Scenario | Data Size | Approx. Time |
|----------|-----------|-------------|
| Sentinel-2 tile (10980 × 10980 px) | ~500 MB stack | 4–8 min |
| Clipped AOI (2000 × 2000 px) | ~40 MB | 30–60 s |
| Drone orthomosaic (5 cm GSD, 1 ha) | ~150 MB | 2–4 min |
| Drone orthomosaic (5 cm GSD, 10 ha) | ~1.5 GB | 15–25 min |

---

## 7. Streamlit Application

The full interactive pipeline is accessible via:

```bash
streamlit run app.py
```

The 6-page workflow covers: Ingestion → Preprocessing → Feature Engineering →
Classification → Post-processing → Export & Report.

---

## 8. Data Sources and Licensing

| Dataset | Source | Licence |
|---------|--------|---------|
| Sentinel-2 L2A | [Copernicus Data Space](https://dataspace.copernicus.eu/) | Copernicus Open Access — free reuse with attribution |
| Sentinel-1 GRD | Copernicus Data Space | Copernicus Open Access — free reuse with attribution |
| Copernicus DEM GLO-30 | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/) | Copernicus DEM licence (free for commercial and research use) |

Attribution: *Contains modified Copernicus Sentinel data [year].*

---

## 9. References

- ICMM (2022). *Slope Stability Guidelines for Open Pit Mining*
- MAC (2020). *Tailings Storage Facility Guidelines*, Mining Association of Canada
- ANCOLD (2019). *Guidelines on Tailings Dams*
- DMIRS (2021). *Tailings Storage Facilities — Reporting Requirements*, WA Department of Mines
- Mine Safety and Inspection Act 1994 (WA)
- Rouse et al. (1974). *Monitoring vegetation systems in the Great Plains with ERTS*
- Escadafal & Huete (1991). *Improvement in remote sensing of low vegetation cover in arid regions*

---

*TerraForge Mining Intelligence — Automated Geospatial QA for Mining Consulting*
