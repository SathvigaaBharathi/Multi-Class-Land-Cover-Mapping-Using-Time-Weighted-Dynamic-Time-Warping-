"""
01b_download_s1_composites.py
------------------------------
Stage 1b: STAC-based Sentinel-1 GRD tile download and monthly composite generation.

Downloads Sentinel-1 Ground Range Detected (GRD) Interferometric Wide Swath
imagery monthly median composites for a study area defined by a GeoPackage
boundary file, sourced from the Microsoft Planetary Computer (MPC) STAC API.

Outputs one Cloud Optimised GeoTIFF (COG) per tile per month containing
3 bands: [VV_dB, VH_dB, VV-VH_ratio_dB] as float32 in dB units.

Physical conversions applied:
    sigma0_dB    = 10 * log10(max(sigma0_linear, 1e-9))
    ratio_dB     = VV_dB - VH_dB

Run this alongside 01_download_s2_composites.py (can run sequentially or on
separate machines). Both must complete before Stage 2 (02_preprocess_stack.py).

Usage:
    python 01b_download_s1_composites.py

Configure all parameters in the SETTINGS block below before running.
"""

import os
import shutil
import warnings
import gc
from ctypes import CDLL

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 — required for .rio accessor
import rasterio
import geopandas as gpd
import dask
from dask.distributed import Client, LocalCluster
from multiprocessing import freeze_support

import pystac_client
import planetary_computer
import stackstac

from shapely.geometry import box

# ==============================================================================
# SETTINGS — edit these before running
# ==============================================================================
AOI_FILE_PATH      = "Maharashtra_Districts.gpkg"        # Study area boundary GeoPackage
OUTPUT_DIR         = "TILED_COMPOSITES_MAHARASHTRA_S1"   # Output root directory

TARGET_MONTH_INDEX = None   # Set to int (0–6) to process one month, None for all
WORKERS            = 35     # Dask workers (tune to available cores)
THREADS_PER_WORKER = 2
BATCH_SIZE         = 30     # Tiles per MPC authentication batch (~1 hour token window)

TARGET_CRS         = "EPSG:3857"
TILE_SIZE_METERS   = 20_000   # Must match Stage 1 S2 tile size
OVERLAP_METERS     = 250      # Must match Stage 1 S2 overlap

# Sentinel-1 polarisation bands to download
S1_BANDS = ["vv", "vh"]

# Minimum valid backscatter value (linear power) — avoids log10(0)
S1_EPSILON = 1e-9

# Self-healing threshold: tiles above this size are treated as complete.
# Empirically determined for 3-band float32 LZW tiles in this study area.
VALID_TILE_SIZE_BYTES = 500_000  # 500 KB

ALL_MONTHS = [
    ("2023-10-01", "2023-10-31", 10),
    ("2023-11-01", "2023-11-30", 11),
    ("2023-12-01", "2023-12-31", 12),
    ("2024-01-01", "2024-01-31",  1),
    ("2024-02-01", "2024-02-29",  2),
    ("2024-03-01", "2024-03-31",  3),
    ("2024-04-01", "2024-04-30",  4),
]
# ==============================================================================


def get_aoi(file_path: str, target_crs: str):
    """Load study area boundary and return projected geometry."""
    gdf = gpd.read_file(file_path).to_crs(target_crs)
    return gdf.geometry.union_all()


def create_processing_grid(aoi_geom, target_crs: str, tile_size: int, overlap: int):
    """
    Generate a regular tile grid covering the AOI.

    Must use identical parameters to the S2 downloader so tile filenames
    align between Stage 1 (S2) and Stage 1b (S1) outputs.
    """
    minx, miny, maxx, maxy = aoi_geom.bounds
    step = tile_size - overlap
    tiles = []
    for x in np.arange(minx, maxx + step, step):
        for y in np.arange(miny, maxy + step, step):
            tile_box = box(x, y, x + tile_size, y + tile_size)
            if aoi_geom.intersects(tile_box):
                tiles.append(tile_box)
    return gpd.GeoDataFrame(tiles, columns=["geometry"], crs=target_crs)


def query_stac_s1(catalog, tile_geom_4326, time_range: str):
    """
    Query MPC STAC for Sentinel-1 GRD IW items intersecting a tile.

    Filters to Interferometric Wide (IW) swath mode and GRD product type.
    No cloud cover filter applies to SAR data.
    """
    try:
        return catalog.search(
            collections=["sentinel-1-grd"],
            intersects=tile_geom_4326,
            datetime=time_range,
            query={
                "platform": {"in": ["sentinel-1a", "sentinel-1b"]},
                "s1:instrument_mode": {"eq": "IW"},
            },
            limit=100,
        ).item_collection()
    except Exception:
        return []


@dask.delayed
def process_tile(tile_row, month_tiles_dir: str, tile_items):
    """
    Download, composite, and convert a single Sentinel-1 tile for one month.

    Processing steps:
        1. Stack all available GRD scenes for the month
        2. Compute monthly median in linear power space
        3. Convert to dB: sigma0_dB = 10 * log10(max(sigma0_linear, epsilon))
        4. Compute VV-VH polarisation ratio in dB space
        5. Write [VV_dB, VH_dB, ratio_dB] as float32 COG

    Self-healing: tiles above VALID_TILE_SIZE_BYTES are skipped as complete.
    Corrupt/interrupted tiles are consistently below this threshold.
    """
    import rioxarray  # noqa: F401 — required inside Dask worker
    import rasterio

    tile_id  = tile_row.name
    out_path = os.path.join(month_tiles_dir, f"tile_{tile_id:05d}.tif")

    # Self-healing resume check
    if os.path.exists(out_path):
        if os.path.getsize(out_path) > VALID_TILE_SIZE_BYTES:
            return f"Skipped (already complete): tile_{tile_id:05d}.tif"
        try:
            os.remove(out_path)
        except OSError:
            pass

    try:
        signed_items = [planetary_computer.sign(item) for item in tile_items]

        gdal_cache_dir = f"gdal_cache_s1_{tile_id}"
        os.makedirs(gdal_cache_dir, exist_ok=True)

        with rasterio.Env(
            VSI_CURL_CACHE_PATH=gdal_cache_dir,
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            VSI_CACHE="TRUE",
            VSI_CACHE_SIZE="200000000",
        ):
            cube = stackstac.stack(
                signed_items,
                assets=S1_BANDS,
                bounds=tile_row.geometry.bounds,
                epsg=3857,
                dtype="float32",
                chunksize=2048,
                rescale=False,
            )

            # Monthly median in linear power space (before log transform)
            # Median is more robust to speckle outliers than mean
            comp = cube.median(dim="time", skipna=True)

            if comp.isnull().all():
                shutil.rmtree(gdal_cache_dir, ignore_errors=True)
                return f"Skipped (no valid data): tile_{tile_id:05d}"

            comp = comp.astype("float32")

            # Convert linear power to dB
            vv_linear = comp.sel(band="vv")
            vh_linear = comp.sel(band="vh")

            vv_db = (10.0 * np.log10(vv_linear.clip(min=S1_EPSILON))).expand_dims(band=["VV"])
            vh_db = (10.0 * np.log10(vh_linear.clip(min=S1_EPSILON))).expand_dims(band=["VH"])

            # VV - VH ratio in dB space
            ratio_db = (vv_db.squeeze() - vh_db.squeeze()).expand_dims(band=["ratio"])

            final = xr.concat([vv_db, vh_db, ratio_db], dim="band")
            final = (
                final
                .rio.write_crs(cube.crs)
                .rio.clip([tile_row.geometry], all_touched=True, drop=False)
            )
            final.rio.to_raster(out_path, driver="COG", compress="LZW")

        shutil.rmtree(gdal_cache_dir, ignore_errors=True)
        return f"Saved: tile_{tile_id:05d}.tif"

    except Exception as exc:
        return f"Failed (tile_{tile_id:05d}): {exc}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aoi_geom    = get_aoi(AOI_FILE_PATH, TARGET_CRS)
    grid_gdf    = create_processing_grid(aoi_geom, TARGET_CRS, TILE_SIZE_METERS, OVERLAP_METERS)
    grid_gdf_4326 = grid_gdf.to_crs("EPSG:4326")

    months_to_run = (
        ALL_MONTHS if TARGET_MONTH_INDEX is None else [ALL_MONTHS[TARGET_MONTH_INDEX]]
    )

    client = Client(LocalCluster(n_workers=WORKERS, threads_per_worker=THREADS_PER_WORKER))
    print(f"Dask dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    all_rows      = list(grid_gdf.iterrows())
    all_rows_4326 = list(grid_gdf_4326.iterrows())
    num_batches   = int(np.ceil(len(all_rows) / BATCH_SIZE))

    for start_date, end_date, month_num in months_to_run:
        time_range = f"{start_date}/{end_date}"
        month_dir  = os.path.join(OUTPUT_DIR, f"{start_date[:4]}_{month_num:02d}_tiles")
        os.makedirs(month_dir, exist_ok=True)

        print(f"\nProcessing {start_date[:7]} — {len(all_rows)} tiles in {num_batches} batches")

        for i in range(num_batches):
            batch      = all_rows[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            batch_4326 = all_rows_4326[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            tasks = []

            for (_, row), (_, row_4326) in zip(batch, batch_4326):
                out_path = os.path.join(month_dir, f"tile_{row.name:05d}.tif")
                if os.path.exists(out_path) and os.path.getsize(out_path) > VALID_TILE_SIZE_BYTES:
                    continue  # Already complete — skip submission
                items = query_stac_s1(catalog, row_4326.geometry, time_range)
                if items:
                    tasks.append(process_tile(row, month_dir, items))

            if tasks:
                print(f"  Batch {i + 1}/{num_batches}: {len(tasks)} tiles")
                client.compute(tasks, sync=True)
                gc.collect()
                try:
                    CDLL("libc.so.6").malloc_trim(0)  # Linux only — trim glibc heap
                except OSError:
                    pass

    client.close()
    print("\nSentinel-1 download complete.")


if __name__ == "__main__":
    freeze_support()
    main()
