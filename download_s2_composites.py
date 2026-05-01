"""
01_download_s2_composites.py
-----------------------------
Stage 1: STAC-based Sentinel-2 tile download and monthly composite generation.

Downloads Sentinel-2 L2A surface reflectance monthly geometric-median composites
for a study area defined by a GeoPackage boundary file, sourced from the
Microsoft Planetary Computer (MPC) STAC API.

Outputs one Cloud Optimised GeoTIFF (COG) per tile per month containing
5 bands: [B02, B03, B04, B08, NDVI] as float32 in [0, 1] range.

Usage:
    python 01_download_s2_composites.py

Configure all parameters in the SETTINGS block below before running.
"""

import os
import sys
import time
import shutil
import warnings
import gc
from ctypes import CDLL

warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import rioxarray
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
AOI_FILE_PATH       = "Maharashtra_Districts.gpkg"  # Path to district boundary GeoPackage
OUTPUT_DIR          = "TILED_COMPOSITES_MAHARASHTRA_S2"  # Output root directory

TARGET_MONTH_INDEX  = None  # Set to an int (0–6) to process one month, or None for all
WORKERS             = 35    # Dask workers (tune to available cores)
THREADS_PER_WORKER  = 2
BATCH_SIZE          = 30    # Tiles per authentication batch (MPC token expires ~1 hour)
CLOUD_COVER_MAX     = 80    # Maximum scene cloud cover (%) for STAC query

TARGET_CRS          = "EPSG:3857"
TILE_SIZE_METERS    = 20_000   # 20 km tiles
OVERLAP_METERS      = 250      # 250 m overlap to avoid edge artefacts
SAFE_NODATA_VALUE   = -99999

BANDS = ["B02", "B03", "B04", "B08", "SCL"]

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
    """Load study area boundary and return geometry + WGS84 bounds."""
    gdf = gpd.read_file(file_path).to_crs(target_crs)
    return gdf.geometry.union_all(), gdf.to_crs("EPSG:4326").total_bounds


def create_processing_grid(aoi_geom, target_crs: str, tile_size: int, overlap: int):
    """Generate a regular tile grid covering the AOI."""
    minx, miny, maxx, maxy = aoi_geom.bounds
    step = tile_size - overlap
    tiles = []
    for x in np.arange(minx, maxx + step, step):
        for y in np.arange(miny, maxy + step, step):
            tile_box = box(x, y, x + tile_size, y + tile_size)
            if aoi_geom.intersects(tile_box):
                tiles.append(tile_box)
    return gpd.GeoDataFrame(tiles, columns=["geometry"], crs=target_crs)


def query_stac(catalog, tile_geom_4326, time_range: str, cloud_max: int):
    """Query MPC STAC for Sentinel-2 items intersecting a tile."""
    try:
        return catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=tile_geom_4326,
            datetime=time_range,
            query={"eo:cloud_cover": {"lt": cloud_max}},
            limit=100,
        ).item_collection()
    except Exception:
        return []


@dask.delayed
def process_tile(tile_row, month_tiles_dir: str, tile_items):
    """
    Download, cloud-mask, and composite a single tile for one month.

    Self-healing: tiles larger than 3 MB are treated as complete and skipped.
    This threshold is empirically reliable for 56-band float32 LZW tiles
    in this study area; corrupt/interrupted tiles are consistently <3 MB.
    """
    import rioxarray  # noqa: F401 — required inside Dask worker
    import rasterio

    tile_id = tile_row.name
    out_path = os.path.join(month_tiles_dir, f"tile_{tile_id:05d}.tif")

    # Self-healing resume check
    if os.path.exists(out_path):
        if os.path.getsize(out_path) > 3_000_000:
            return f"Skipped (already complete): tile_{tile_id:05d}.tif"
        try:
            os.remove(out_path)
        except OSError:
            pass

    try:
        signed_items = [planetary_computer.sign(item) for item in tile_items]

        gdal_cache_dir = f"gdal_cache_{tile_id}"
        os.makedirs(gdal_cache_dir, exist_ok=True)

        with rasterio.Env(
            VSI_CURL_CACHE_PATH=gdal_cache_dir,
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            VSI_CACHE="TRUE",
            VSI_CACHE_SIZE="200000000",
        ):
            cube = stackstac.stack(
                signed_items,
                assets=BANDS,
                bounds=tile_row.geometry.bounds,
                epsg=3857,
                dtype="int64",
                fill_value=SAFE_NODATA_VALUE,
                chunksize=2048,
                rescale=False,
            )

            # Cloud masking via SCL: exclude shadows (3), medium/high clouds (8,9),
            # thin cirrus (10), snow/ice (11)
            scl_mask = cube.sel(band="SCL").isin([3, 8, 9, 10, 11])
            masked = cube.where(~scl_mask)

            # Monthly geometric-median composite
            comp = masked.sel(band=["B02", "B03", "B04", "B08"]).median(
                dim="time", skipna=True
            )

            if comp.isnull().all():
                shutil.rmtree(gdal_cache_dir, ignore_errors=True)
                return f"Skipped (no valid data): tile_{tile_id:05d}"

            # Convert DN to surface reflectance: (DN - 1000) / 10000
            comp_sr = (comp.astype("float32") - 1000.0) / 10000.0

            # Compute NDVI
            nir = comp_sr.sel(band="B08")
            red = comp_sr.sel(band="B04")
            ndvi = ((nir - red) / (nir + red + 1e-9)).expand_dims(band=["NDVI"])

            final = xr.concat([comp_sr, ndvi], dim="band")
            final = (
                final
                .rio.write_crs(cube.crs)
                .rio.clip([tile_row.geometry], all_touched=True, drop=False)
            )
            final.rio.to_raster(out_path, driver="COG", compress="LZW",
                                nodata=SAFE_NODATA_VALUE)

        shutil.rmtree(gdal_cache_dir, ignore_errors=True)
        return f"Saved: tile_{tile_id:05d}.tif"

    except Exception as exc:
        return f"Failed (tile_{tile_id:05d}): {exc}"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    aoi_geom, _ = get_aoi(AOI_FILE_PATH, TARGET_CRS)
    grid_gdf = create_processing_grid(aoi_geom, TARGET_CRS, TILE_SIZE_METERS, OVERLAP_METERS)
    grid_gdf_4326 = grid_gdf.to_crs("EPSG:4326")

    months_to_run = ALL_MONTHS if TARGET_MONTH_INDEX is None else [ALL_MONTHS[TARGET_MONTH_INDEX]]

    client = Client(LocalCluster(n_workers=WORKERS, threads_per_worker=THREADS_PER_WORKER))
    print(f"Dask dashboard: {client.dashboard_link}")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )

    all_rows     = list(grid_gdf.iterrows())
    all_rows_4326 = list(grid_gdf_4326.iterrows())
    num_batches  = int(np.ceil(len(all_rows) / BATCH_SIZE))

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
                if os.path.exists(out_path) and os.path.getsize(out_path) > 3_000_000:
                    continue  # Already complete — skip submission
                items = query_stac(catalog, row_4326.geometry, time_range, CLOUD_COVER_MAX)
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
    print("\nDownload complete.")


if __name__ == "__main__":
    freeze_support()
    main()
