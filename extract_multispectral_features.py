
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio


# ============================================================
# SPECTRAAI — MULTISPECTRAL FEATURE EXTRACTION
# Dataset: NASA HLS S30 v2.0
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "HLS_S30"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"

OUTPUT_FILE = OUTPUT_DIR / "multispectral_features.csv"

BANDS = ["B02", "B03", "B04", "B8A"]
ALL_LAYERS = BANDS + ["Fmask"]

FILENAME_PATTERN = re.compile(
    r"HLSS30\.020_(B02|B03|B04|B8A|Fmask)_"
    r"doy(\d{7})_aid(\d{4})_(\d{2}N)\.tif$",
    re.IGNORECASE
)

# Fmask bit definitions from the official HLS lookup
CLOUD_BIT = 2
ADJACENT_BIT = 4
SHADOW_BIT = 8
SNOW_BIT = 16
WATER_BIT = 32

# Aerosol classes
AEROSOL_LOW = 64
AEROSOL_MODERATE = 128
AEROSOL_HIGH = 192

NODATA_FMASK = 255


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_stats(values, prefix):
    """Calculate mean and standard deviation."""

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values))
    }


def calculate_ratio(numerator, denominator, mask):
    """Calculate a spectral ratio while avoiding division by zero."""

    result = np.full(denominator.shape, np.nan, dtype=np.float32)

    valid = mask & np.isfinite(numerator) & np.isfinite(denominator)
    valid &= np.abs(denominator) > 1e-6

    np.divide(
        numerator,
        denominator,
        out=result,
        where=valid
    )

    return result


def validate_grid(datasets):
    """Ensure all five raster layers share the same spatial grid."""

    reference = datasets[0]

    for dataset in datasets[1:]:
        if dataset.width != reference.width:
            raise ValueError("Raster width mismatch.")

        if dataset.height != reference.height:
            raise ValueError("Raster height mismatch.")

        if dataset.crs != reference.crs:
            raise ValueError("Raster CRS mismatch.")

        if dataset.transform != reference.transform:
            raise ValueError("Raster transform mismatch.")


def read_reflectance(path):
    """Read a reflectance band and apply its stored scale and offset."""

    with rasterio.open(path) as src:
        raw = src.read(1)

        nodata = src.nodata
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0

        if scale is None:
            scale = 1.0

        if offset is None:
            offset = 0.0

        valid = np.isfinite(raw)

        if nodata is not None:
            valid &= raw != nodata

        reflectance = (
            raw.astype(np.float32) * float(scale)
            + float(offset)
        )

        reflectance[~valid] = np.nan

        return reflectance, valid


# ============================================================
# SPECTRAL INDEX CALCULATIONS
# ============================================================

def calculate_spectral_features(bands, mask, prefix):

    blue = bands["B02"]
    green = bands["B03"]
    red = bands["B04"]
    nir = bands["B8A"]

    features = {}

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = calculate_ratio(
        nir - red,
        nir + red,
        mask
    )

    # NDWI = (Green - NIR) / (Green + NIR)
    ndwi = calculate_ratio(
        green - nir,
        green + nir,
        mask
    )

    # EVI = 2.5 * (NIR - Red) /
    #       (NIR + 6*Red - 7.5*Blue + 1)
    evi_denominator = (
        nir + 6 * red - 7.5 * blue + 1
    )

    evi = calculate_ratio(
        2.5 * (nir - red),
        evi_denominator,
        mask
    )

    features.update(
        calculate_stats(
            ndvi[mask],
            f"{prefix}_NDVI"
        )
    )

    features.update(
        calculate_stats(
            ndwi[mask],
            f"{prefix}_NDWI"
        )
    )

    features.update(
        calculate_stats(
            evi[mask],
            f"{prefix}_EVI"
        )
    )

    return features


# ============================================================
# IMAGE QUALITY FEATURES
# ============================================================

def calculate_image_quality(bands, clear_mask):

    features = {}

    red = bands["B04"]
    green = bands["B03"]
    blue = bands["B02"]

    # Approximate visible brightness using RGB reflectance.
    brightness = (
        0.2126 * red
        + 0.7152 * green
        + 0.0722 * blue
    )

    features.update(
        calculate_stats(
            brightness[clear_mask],
            "clear_brightness"
        )
    )

    # Contrast proxy: standard deviation of visible brightness.
    features["clear_contrast"] = (
        float(np.std(brightness[clear_mask]))
        if np.any(clear_mask)
        else np.nan
    )

    # Spatial sharpness proxy using the red band.
    # A percentile stretch makes the Laplacian calculation
    # less dependent on the original reflectance scale.

    valid_red = red[clear_mask]
    valid_red = valid_red[np.isfinite(valid_red)]

    if valid_red.size > 20:

        low, high = np.percentile(valid_red, [2, 98])

        if high > low:

            stretched = np.zeros(red.shape, dtype=np.uint8)

            normalized = np.clip(
                (red - low) / (high - low),
                0,
                1
            )

            normalized = np.nan_to_num(
                normalized,
                nan=0.0
            )

            stretched = (normalized * 255).astype(np.uint8)

            # Exclude pixels near masked boundaries.
            kernel = np.ones((3, 3), dtype=np.uint8)

            safe_mask = cv2.erode(
                clear_mask.astype(np.uint8),
                kernel,
                iterations=1
            ).astype(bool)

            laplacian = cv2.Laplacian(
                stretched,
                cv2.CV_64F
            )

            if np.any(safe_mask):
                features["red_laplacian_variance"] = float(
                    np.var(laplacian[safe_mask])
                )
            else:
                features["red_laplacian_variance"] = np.nan

        else:
            features["red_laplacian_variance"] = np.nan

    else:
        features["red_laplacian_variance"] = np.nan

    return features


# ============================================================
# FMASK QUALITY ANALYSIS
# ============================================================

def calculate_fmask_features(fmask):

    features = {}

    total_pixels = fmask.size

    valid_fmask = (
        (fmask != NODATA_FMASK)
        & np.isfinite(fmask)
    )

    valid_count = int(np.count_nonzero(valid_fmask))

    features["total_pixels"] = int(total_pixels)
    features["fmask_valid_pixels"] = valid_count

    features["fmask_nodata_pct"] = (
        100 * (total_pixels - valid_count) / total_pixels
        if total_pixels else np.nan
    )

    if valid_count == 0:
        return features, valid_fmask, np.zeros(
            fmask.shape, dtype=bool
        )

    valid_values = fmask[valid_fmask].astype(np.uint8)

    def flag_percentage(bit):
        flagged = (valid_values & bit) != 0
        return float(100 * np.count_nonzero(flagged) / valid_count)

    features["cloud_pct"] = flag_percentage(CLOUD_BIT)
    features["adjacent_cloud_shadow_pct"] = flag_percentage(
        ADJACENT_BIT
    )
    features["cloud_shadow_pct"] = flag_percentage(SHADOW_BIT)
    features["snow_ice_pct"] = flag_percentage(SNOW_BIT)
    features["water_pct"] = flag_percentage(WATER_BIT)

    # Aerosol base is encoded in the upper two bits.
    aerosol_classes = valid_values & 192

    features["low_aerosol_pct"] = float(
        100 * np.count_nonzero(
            aerosol_classes == AEROSOL_LOW
        ) / valid_count
    )

    features["moderate_aerosol_pct"] = float(
        100 * np.count_nonzero(
            aerosol_classes == AEROSOL_MODERATE
        ) / valid_count
    )

    features["high_aerosol_pct"] = float(
        100 * np.count_nonzero(
            aerosol_classes == AEROSOL_HIGH
        ) / valid_count
    )

    # Clear means no cloud, shadow, snow or adjacency flag.
    # Water is deliberately retained.
    excluded_flags = (
        CLOUD_BIT
        | ADJACENT_BIT
        | SHADOW_BIT
        | SNOW_BIT
    )

    clear_mask = valid_fmask & (
        (fmask.astype(np.uint8) & excluded_flags) == 0
    )

    clear_count = int(np.count_nonzero(clear_mask))

    features["clear_pixels_pct"] = (
        100 * clear_count / valid_count
    )

    return features, valid_fmask, clear_mask


# ============================================================
# PROCESS ONE OBSERVATION
# ============================================================

def process_observation(key, files):

    date, area_id, tile = key

    missing = [
        layer for layer in ALL_LAYERS
        if layer not in files
    ]

    if missing:
        raise ValueError(
            f"Missing layers: {', '.join(missing)}"
        )

    with rasterio.open(files["Fmask"]) as fmask_src:

        fmask = fmask_src.read(1)

        with rasterio.open(files["B02"]) as b02_src, \
             rasterio.open(files["B03"]) as b03_src, \
             rasterio.open(files["B04"]) as b04_src, \
             rasterio.open(files["B8A"]) as b8a_src:

            validate_grid([
                b02_src,
                b03_src,
                b04_src,
                b8a_src,
                fmask_src
            ])

            bands = {}
            band_masks = {}

            for band, src in [
                ("B02", b02_src),
                ("B03", b03_src),
                ("B04", b04_src),
                ("B8A", b8a_src)
            ]:

                raw = src.read(1)

                nodata = src.nodata
                scale = src.scales[0] if src.scales else 1.0
                offset = src.offsets[0] if src.offsets else 0.0

                if scale is None:
                    scale = 1.0

                if offset is None:
                    offset = 0.0

                valid = np.isfinite(raw)

                if nodata is not None:
                    valid &= raw != nodata

                reflectance = (
                    raw.astype(np.float32) * float(scale)
                    + float(offset)
                )

                reflectance[~valid] = np.nan

                bands[band] = reflectance
                band_masks[band] = valid

    total_pixels = fmask.size

    # Require all four reflectance bands and Fmask to be valid.
    common_valid = (
        band_masks["B02"]
        & band_masks["B03"]
        & band_masks["B04"]
        & band_masks["B8A"]
        & (fmask != NODATA_FMASK)
    )

    row = {
        "date_doy": date,
        "area_id": area_id,
        "tile": tile,
        "total_pixels": int(total_pixels)
    }

    row["reflectance_valid_pixels"] = int(
        np.count_nonzero(common_valid)
    )

    row["reflectance_valid_coverage_pct"] = (
        100 * np.count_nonzero(common_valid) / total_pixels
        if total_pixels else np.nan
    )

    # --------------------------------------------------------
    # Reflectance statistics: all valid pixels
    # --------------------------------------------------------

    for band in BANDS:
        row.update(
            calculate_stats(
                bands[band][common_valid],
                f"all_valid_{band}"
            )
        )

    # --------------------------------------------------------
    # Fmask statistics
    # --------------------------------------------------------

    fmask_features, fmask_valid, clear_mask = (
        calculate_fmask_features(fmask)
    )

    row.update(fmask_features)

    # Only use pixels valid in every reflectance band and
    # classified as clear by Fmask for clear spectral metrics.
    clear_reflectance_mask = common_valid & clear_mask

    row["clear_reflectance_pixels"] = int(
        np.count_nonzero(clear_reflectance_mask)
    )

    row["clear_reflectance_coverage_pct"] = (
        100 * np.count_nonzero(clear_reflectance_mask)
        / total_pixels
        if total_pixels else np.nan
    )

    # --------------------------------------------------------
    # Clear-pixel reflectance statistics
    # --------------------------------------------------------

    for band in BANDS:
        row.update(
            calculate_stats(
                bands[band][clear_reflectance_mask],
                f"clear_{band}"
            )
        )

    # --------------------------------------------------------
    # Spectral indices
    # --------------------------------------------------------

    row.update(
        calculate_spectral_features(
            bands,
            clear_reflectance_mask,
            "clear"
        )
    )

    # --------------------------------------------------------
    # Image quality proxies
    # --------------------------------------------------------

    row.update(
        calculate_image_quality(
            bands,
            clear_reflectance_mask
        )
    )

    return row


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nSPECTRAAI — MULTISPECTRAL FEATURE EXTRACTION")
    print("=" * 55)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group files by date, area and tile.
    observations = {}

    tif_files = list(DATA_DIR.glob("*.tif"))

    for path in tif_files:

        match = FILENAME_PATTERN.match(path.name)

        if not match:
            continue

        layer, date, area_id, tile = match.groups()

        key = (date, area_id, tile)

        observations.setdefault(key, {})

        if layer in observations[key]:
            raise ValueError(
                f"Duplicate layer {layer} for observation {key}"
            )

        observations[key][layer] = path

    print(f"Total TIFF files found: {len(tif_files)}")
    print(f"Observations discovered: {len(observations)}")

    results = []
    errors = []

    for index, (key, files) in enumerate(
        sorted(observations.items()),
        start=1
    ):

        print(
            f"\nProcessing {index}/{len(observations)}: "
            f"{key[0]} | Area {key[1]} | Tile {key[2]}"
        )

        try:
            row = process_observation(key, files)
            results.append(row)

            print("  SUCCESS")

        except Exception as error:
            errors.append({
                "date_doy": key[0],
                "area_id": key[1],
                "tile": key[2],
                "error": str(error)
            })

            print(f"  ERROR: {error}")

    if results:

        dataframe = pd.DataFrame(results)

        dataframe = dataframe.sort_values(
            ["date_doy", "area_id", "tile"]
        )

        dataframe.to_csv(
            OUTPUT_FILE,
            index=False,
            float_format="%.8f"
        )

        print("\n" + "=" * 55)
        print("EXTRACTION SUMMARY")
        print("=" * 55)

        print(f"Observations discovered: {len(observations)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed observations: {len(errors)}")
        print(f"Features generated: {len(dataframe.columns)}")

        print("\nOutput saved:")
        print(OUTPUT_FILE)

        print("\nPreview:")
        print(
            dataframe[
                [
                    "date_doy",
                    "area_id",
                    "tile",
                    "clear_pixels_pct",
                    "high_aerosol_pct",
                    "clear_NDVI_mean",
                    "red_laplacian_variance"
                ]
            ].head(10).to_string(index=False)
        )

    if errors:

        print("\nFAILED OBSERVATIONS")

        for error in errors:
            print(error)

    if not results:
        print("\nNo observations were successfully processed.")

    print("\nFeature extraction finished.")


if __name__ == "__main__":
    main()
