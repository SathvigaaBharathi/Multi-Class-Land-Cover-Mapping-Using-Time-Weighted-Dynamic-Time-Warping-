"""
02_preprocess_stack.py
-----------------------
Stage 2: Multi-sensor preprocessing and temporal feature stacking.

Reads monthly Sentinel-2 (S2) and Sentinel-1 (S1) tile directories,
applies physical unit conversions, computes spectral/polarimetric indices,
and writes a single 56-band float32 GeoTIFF per tile containing the full
temporal feature stack used as TWDTW classifier input.

Band layout per tile (56 bands total = 7 months × 8 bands/month):
    Per month: [B02, B03, B04, B08, NDVI, VV_dB, VH_dB, VV-VH_ratio_dB]

Physical conversions applied:
    S2: surface reflectance = (raw_DN - 1000) / 10000, clipped to [0, 1]
    S1: sigma0_dB = 10 * log10(max(sigma0_linear, 1e-9))
    S1 ratio: ratio_dB = VV_dB - VH_dB

NOTE: No per-pixel normalisation is applied. Raw physical values are
preserved intentionally — see paper Section III-B for justification.

Usage:
    python 02_preprocess_stack.py
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

# ==============================================================================
# SETTINGS
# ==============================================================================
DIR_S2     = "TILED_COMPOSITES_MAHARASHTRA_S2"   # Output of Stage 1 (S2 download)
DIR_S1     = "TILED_COMPOSITES_MAHARASHTRA_S1"   # Output of S1 download (same structure)
OUTPUT_DIR = "TILED_COMPOSITES_MAHARASHTRA_FINAL"  # Stacked output tiles

MONTHS = [
    "2023_10_tiles",
    "2023_11_tiles",
    "2023_12_tiles",
    "2024_01_tiles",
    "2024_02_tiles",
    "2024_03_tiles",
    "2024_04_tiles",
]

# S2 band indices (1-based, as stored in Stage 1 output): B02, B03, B04, B08
S2_BAND_INDICES = [1, 2, 3, 4]
# S1 band indices (1-based): VV, VH
S1_BAND_INDICES = [1, 2]

TOTAL_OUTPUT_BANDS = 56  # 7 months * 8 bands
# ==============================================================================


def read_and_align(
    path: str,
    target_meta: dict,
    bands_to_read: list,
) -> np.ndarray:
    """
    Read a raster and reproject/resample it to match target_meta grid.

    Returns a zero-filled array of shape (len(bands_to_read), H, W) if
    the file does not exist or reprojection fails.
    """
    h, w = target_meta["height"], target_meta["width"]
    fallback = np.zeros((len(bands_to_read), h, w), dtype=np.float32)

    if not os.path.exists(path):
        return fallback

    try:
        with rasterio.open(path) as src:
            dst_shape = (len(bands_to_read), h, w)
            destination = np.zeros(dst_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, bands_to_read),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_meta["transform"],
                dst_crs=target_meta["crs"],
                resampling=Resampling.nearest,
            )
            return destination
    except Exception as exc:
        print(f"\nWarp failed for {os.path.basename(path)}: {exc} — filling with zeros")
        return fallback


def apply_s2_physics(raw: np.ndarray) -> np.ndarray:
    """
    Convert S2 DN to surface reflectance and append NDVI.

    Input shape:  (4, H, W) — bands [B02, B03, B04, B08] as raw DN
    Output shape: (5, H, W) — [B02, B03, B04, B08, NDVI] in [0, 1]
    """
    sr = (raw - 1000.0) / 10000.0
    sr = np.clip(sr, 0.0, 1.0)

    red, nir = sr[2], sr[3]
    ndvi = np.divide(
        nir - red,
        nir + red,
        out=np.zeros_like(nir),
        where=(nir + red) != 0,
    )
    return np.concatenate([sr, ndvi[np.newaxis]], axis=0)  # (5, H, W)


def apply_s1_physics(raw: np.ndarray) -> np.ndarray:
    """
    Convert S1 linear power to dB and append VV-VH ratio.

    Input shape:  (2, H, W) — [VV_linear, VH_linear]
    Output shape: (3, H, W) — [VV_dB, VH_dB, ratio_dB]
    """
    db = 10.0 * np.log10(np.maximum(raw, 1e-9))
    ratio = (db[0] - db[1])[np.newaxis]  # (1, H, W)
    return np.concatenate([db, ratio], axis=0)  # (3, H, W)


def process_workflow():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    master_s2_dir = os.path.join(DIR_S2, MONTHS[0])
    if not os.path.exists(master_s2_dir):
        print(f"ERROR: Input directory not found: {master_s2_dir}")
        print("Check DIR_S2 setting and ensure Stage 1 completed successfully.")
        return

    all_tiles = [f for f in os.listdir(master_s2_dir) if f.endswith(".tif")]
    print(f"Found {len(all_tiles)} tiles. Stacking {len(MONTHS)} months "
          f"({TOTAL_OUTPUT_BANDS} bands per tile)...")

    for tile_name in tqdm(all_tiles, desc="Stacking tiles"):
        out_path = os.path.join(OUTPUT_DIR, tile_name)

        # Resume: skip if output exists and is non-empty
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue

        # Establish master grid from the first available S2 month for this tile
        master_meta = None
        for month in MONTHS:
            candidate = os.path.join(DIR_S2, month, tile_name)
            if os.path.exists(candidate):
                try:
                    with rasterio.open(candidate) as src:
                        master_meta = src.meta.copy()
                    break
                except Exception:
                    continue

        if master_meta is None:
            continue  # Tile not present in any month — skip

        tile_stack = []

        for month in MONTHS:
            s2_path = os.path.join(DIR_S2, month, tile_name)
            s1_path = os.path.join(DIR_S1, month, tile_name)

            # Load, align, apply physics
            s2_raw    = read_and_align(s2_path, master_meta, S2_BAND_INDICES)
            s2_final  = apply_s2_physics(s2_raw)   # (5, H, W)

            s1_raw    = read_and_align(s1_path, master_meta, S1_BAND_INDICES)
            s1_final  = apply_s1_physics(s1_raw)   # (3, H, W)

            # Concatenate S2 + S1 for this month: (8, H, W)
            month_stack = np.concatenate([s2_final, s1_final], axis=0)
            tile_stack.append(month_stack)

        if not tile_stack:
            continue

        # Final array: (56, H, W) — all months concatenated along band axis
        final_array = np.concatenate(tile_stack, axis=0).astype(np.float32)

        master_meta.update(
            count=TOTAL_OUTPUT_BANDS,
            dtype=rasterio.float32,
            compress="lzw",
            BIGTIFF="YES",
        )

        try:
            with rasterio.open(out_path, "w", **master_meta) as dst:
                dst.write(final_array)
        except Exception as exc:
            print(f"\nFailed to write {tile_name}: {exc}")

    print(f"\nStacking complete. Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_workflow()
