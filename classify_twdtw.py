"""
03_classify_twdtw.py
---------------------
Stage 3: Parallelised Time-Weighted Dynamic Time Warping (TWDTW) classification.

Reads stacked 56-band tiles from Stage 2, extracts per-class reference
signatures from training polygons, and classifies each pixel using a
Numba JIT-compiled TWDTW kernel.

TWDTW temporal weighting (Maus et al., 2016):
    omega(delta_t) = 1 / (1 + exp(-alpha * (delta_t - beta)))

Parameters:
    beta  = inflection point in month units (default 2.0 = ~2 months)
    alpha = penalty steepness (default 0.5)

Water pre-mask: pixels with mean NDWI > 0 across all months are forced
to the Water class before TWDTW distance computation.

Output classes (configurable via training polygon 'layer' field):
    1 = Cropland, 2 = Forest, 3 = Water, 4 = Settlements, 5 = Barrenland
    (exact mapping depends on training data — Water is also set by NDWI mask)

Usage:
    python 03_classify_twdtw.py
"""

import os
import glob
import multiprocessing

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.mask import mask
from rasterio.merge import merge
from numba import jit, float64, int64
from tqdm import tqdm

# Suppress NumPy/BLAS thread-level parallelism to avoid contention with
# multiprocessing.Pool workers
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["VECLIB_MAXIMUM_THREADS"]  = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"
os.environ["NUMBA_NUM_THREADS"]       = "1"

# ==============================================================================
# SETTINGS
# ==============================================================================
TILES_DIR          = "TILED_COMPOSITES_MAHARASHTRA_FINAL"  # Output of Stage 2
POLYGON_FILE       = "training_polygons.geojson"           # Training polygons (GeoJSON/GPKG)
POLYGON_CLASS_FIELD = "layer"                              # Field containing class label

CLASSIFIED_DIR     = "classified_tiles"
FINAL_OUTPUT_FILE  = "classified_map_maharashtra.tif"

# TWDTW temporal parameters
BETA  = 2.0   # Sigmoid inflection point (months)
ALPHA = 0.5   # Sigmoid slope

# Feature stack dimensions
NUM_MONTHS       = 7
BANDS_PER_MONTH  = 8   # [B02, B03, B04, B08, NDVI, VV_dB, VH_dB, ratio_dB]
TOTAL_BANDS      = NUM_MONTHS * BANDS_PER_MONTH  # 56

# NDWI band indices within each monthly 8-band chunk (0-based)
IDX_GREEN = 1   # B03
IDX_NIR   = 3   # B08

# Number of CPU cores to use (reserves 4 cores for OS and memory management)
NUM_WORKERS = max(1, os.cpu_count() - 4)
# ==============================================================================


# ==============================================================================
# NUMBA TWDTW KERNEL
# ==============================================================================

@jit(float64(float64[:], float64[:]), nopython=True, cache=True)
def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two feature vectors."""
    return np.sqrt(np.sum((a - b) ** 2))


@jit(
    float64(float64[:, :], float64[:, :], float64[:], float64[:], float64, float64),
    nopython=True,
    cache=True,
)
def _twdtw(s1, s2, t1, t2, beta: float, alpha: float) -> float:
    """
    Time-Weighted Dynamic Time Warping distance between two multi-band time series.

    Args:
        s1, s2   : Time series arrays of shape (T, B) — T timesteps, B bands
        t1, t2   : Timestamp arrays of shape (T,)
        beta     : Logistic sigmoid inflection point
        alpha    : Logistic sigmoid slope

    Returns:
        Scalar TWDTW distance (accumulated cost at final alignment cell)
    """
    n, m = len(s1), len(s2)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delta_t = abs(t1[i - 1] - t2[j - 1])
            omega   = 1.0 / (1.0 + np.exp(-alpha * (delta_t - beta)))
            cost    = _euclidean(s1[i - 1], s2[j - 1]) + omega
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return dtw[n, m]


@jit(
    int64[:](float64[:, :, :], float64[:, :, :], float64[:], float64, float64, int64[:]),
    nopython=True,
    cache=True,
)
def _classify_pixels(
    pixels,
    ref_sigs,
    timestamps,
    beta: float,
    alpha: float,
    water_mask,
) -> np.ndarray:
    """
    Classify all valid pixels in a tile via 1-nearest-neighbour TWDTW.

    Args:
        pixels     : (N, T, B) array of pixel time series
        ref_sigs   : (C, T, B) array of per-class reference signatures
        timestamps : (T,) array of month indices
        beta, alpha: TWDTW temporal parameters
        water_mask : (N,) int array — 1 forces pixel to Water class

    Returns:
        (N,) int64 array of class labels (1-based)
    """
    n_pix = pixels.shape[0]
    n_cls = ref_sigs.shape[0]
    out   = np.zeros(n_pix, dtype=np.int64)

    for i in range(n_pix):
        if water_mask[i] == 1:
            out[i] = 3  # Water class (update index to match your class ordering)
            continue

        best_dist  = np.inf
        best_class = -1

        for j in range(n_cls):
            d = _twdtw(pixels[i], ref_sigs[j], timestamps, timestamps, beta, alpha)
            if d < best_dist:
                best_dist  = d
                best_class = j

        out[i] = best_class + 1  # Convert to 1-based class index

    return out


# ==============================================================================
# TILE WORKER (runs in subprocess)
# ==============================================================================

def _classify_tile(args: tuple):
    """
    Worker function: classify a single tile and write result to disk.

    Returns the output path on success, None on failure.
    """
    tile_path, reference_signatures, timestamps, beta, alpha = args
    tile_name     = os.path.basename(tile_path)
    output_path   = os.path.join(CLASSIFIED_DIR, f"classified_{tile_name}")
    class_names   = sorted(reference_signatures.keys())

    if os.path.exists(output_path):
        return output_path  # Already classified — resume

    try:
        with rasterio.open(tile_path) as src:
            img     = src.read()
            profile = src.profile
    except Exception as exc:
        print(f"Read error ({tile_name}): {exc}")
        return None

    img = np.nan_to_num(img, nan=0.0).transpose(1, 2, 0)  # (H, W, 56)
    h, w, bands = img.shape

    if bands != TOTAL_BANDS:
        print(f"Band mismatch in {tile_name}: expected {TOTAL_BANDS}, got {bands}")
        return None

    output = np.zeros((h, w), dtype=np.uint8)
    valid  = np.any(img != 0, axis=2)  # Valid pixel mask

    if not np.any(valid):
        profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(output, 1)
        return output_path

    # Reshape valid pixels to (N, T, B)
    pixels = (
        img[valid]
        .reshape(-1, NUM_MONTHS, BANDS_PER_MONTH)
        .astype(np.float64)
    )

    # NDWI water pre-mask: mean NDWI > 0 over all months → Water
    green = pixels[:, :, IDX_GREEN]
    nir   = pixels[:, :, IDX_NIR]
    ndwi  = (green - nir) / (green + nir + 1e-6)
    water_mask = (np.mean(ndwi, axis=1) > 0.0).astype(np.int64)

    # Build reference signature array (C, T, B)
    ref_arr = np.stack(
        [reference_signatures[name] for name in class_names], axis=0
    ).astype(np.float64)

    # Classify
    labels = _classify_pixels(pixels, ref_arr, timestamps, beta, alpha, water_mask)
    output[valid] = labels

    profile.update(dtype=rasterio.uint8, count=1, nodata=0, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output, 1)

    return output_path


# ==============================================================================
# SIGNATURE EXTRACTION
# ==============================================================================

def extract_signatures(
    polygons: gpd.GeoDataFrame,
    tile_files: list,
    class_field: str,
) -> dict:
    """
    Extract per-class median temporal signatures from training polygons.

    For each class, all polygon pixels are extracted across intersecting tiles,
    concatenated, and the median across pixels is taken to form a (T, B) signature.
    Median is used over mean to suppress mixed-boundary pixels and cloud residuals.
    """
    signatures = {}
    classes    = sorted(polygons[class_field].unique())

    with rasterio.open(tile_files[0]) as src:
        target_crs = src.crs

    if polygons.crs != target_crs:
        polygons = polygons.to_crs(target_crs)

    print("Extracting reference signatures from training polygons...")
    for cls in classes:
        cls_polys = polygons[polygons[class_field] == cls]
        all_pixels = []

        for tf in tile_files:
            with rasterio.open(tf) as src:
                tile_bbox = box(*src.bounds)
                if not cls_polys.geometry.intersects(tile_bbox).any():
                    continue
                for _, row in cls_polys.iterrows():
                    try:
                        out, _ = mask(src, [row.geometry], crop=True)
                        px = np.nan_to_num(out, nan=0.0).reshape(TOTAL_BANDS, -1).T
                        px = px[np.any(px != 0, axis=1)]  # Remove nodata
                        if px.size > 0:
                            all_pixels.append(px)
                    except Exception:
                        continue

        if all_pixels:
            raw = np.vstack(all_pixels).reshape(-1, NUM_MONTHS, BANDS_PER_MONTH)
            signatures[cls] = np.median(raw, axis=0)  # (T, B)
            print(f"  Class '{cls}': signature from {raw.shape[0]} pixels")
        else:
            print(f"  WARNING: No pixels found for class '{cls}' — skipped")

    return signatures


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def classify_workflow():
    os.makedirs(CLASSIFIED_DIR, exist_ok=True)

    tile_files = sorted(glob.glob(os.path.join(TILES_DIR, "*.tif")))
    if not tile_files:
        print(f"No tiles found in: {TILES_DIR}")
        return

    print(f"Found {len(tile_files)} tiles.")

    polygons = gpd.read_file(POLYGON_FILE)
    signatures = extract_signatures(polygons, tile_files, POLYGON_CLASS_FIELD)

    if not signatures:
        print("No signatures extracted. Check polygon/tile overlap and class field name.")
        return

    timestamps = np.arange(NUM_MONTHS, dtype=np.float64)

    tasks = [
        (tf, signatures, timestamps, BETA, ALPHA)
        for tf in tile_files
    ]

    print(f"\nClassifying {len(tasks)} tiles using {NUM_WORKERS} CPU cores...")
    classified_files = []

    with multiprocessing.get_context("spawn").Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(_classify_tile, tasks),
            total=len(tasks),
            desc="Classifying",
        ):
            if result:
                classified_files.append(result)

    if not classified_files:
        print("No tiles were classified successfully.")
        return

    print("\nMerging classified tiles into final mosaic...")
    src_handles = [rasterio.open(f) for f in classified_files]
    mosaic, transform = merge(src_handles)

    meta = src_handles[0].meta.copy()
    meta.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        dtype=rasterio.uint8,
        compress="lzw",
        BIGTIFF="YES",
    )

    with rasterio.open(FINAL_OUTPUT_FILE, "w", **meta) as dst:
        dst.write(mosaic)

    for src in src_handles:
        src.close()

    print(f"\nClassification complete: {FINAL_OUTPUT_FILE}")


if __name__ == "__main__":
    classify_workflow()
