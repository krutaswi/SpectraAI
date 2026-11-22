
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "raw" / "HLS_S30"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"

LOOKUP_FILE = METADATA_DIR / "HLSS30-020-Fmask-lookup.csv"

OUTPUT_FILE = METADATA_DIR / "fmask_quality_metrics.csv"

NODATA_VALUE = 255


# --------------------------------------------------
# LOAD OFFICIAL FMASK LOOKUP TABLE
# --------------------------------------------------

lookup = pd.read_csv(LOOKUP_FILE)

lookup["Value"] = pd.to_numeric(lookup["Value"])

lookup_values = set(lookup["Value"].astype(int))

print("\nSPECTRAAI — FMASK QUALITY ANALYSIS")
print("=" * 55)

print(f"Loaded official lookup entries: {len(lookup)}")


# Find pixel values belonging to each category.

def values_for(column, expected_value):
    return set(
        lookup.loc[
            lookup[column].astype(str).str.strip().str.lower()
            == expected_value.lower(),
            "Value"
        ].astype(int)
    )


cloud_values = values_for("Cloud", "Yes")
shadow_values = values_for("Cloud shadow", "Yes")
snow_values = values_for("Snow/ice", "Yes")
adjacent_values = values_for("Adjacent to cloud/shadow", "Yes")
water_values = values_for("Water", "Yes")

aerosol_values = {
    level: set(
        lookup.loc[
            lookup["Aerosol level"].astype(str).str.strip().str.lower()
            == level.lower(),
            "Value"
        ].astype(int)
    )
    for level in ["Low aerosol", "Moderate aerosol", "High aerosol"]
}


# --------------------------------------------------
# FIND OBSERVATIONS
# --------------------------------------------------

pattern = re.compile(
    r"HLSS30\.020_Fmask_"
    r"doy(\d{7})_aid(\d+)_([A-Za-z0-9]+)\.tif$",
    re.IGNORECASE
)

observations = {}

for path in DATA_DIR.glob("*.tif"):
    match = pattern.match(path.name)

    if not match:
        continue

    date, area_id, tile = match.groups()

    key = (date, area_id, tile)
    observations[key] = path


if not observations:
    raise RuntimeError("No Fmask files were found.")


# --------------------------------------------------
# ANALYZE EACH OBSERVATION
# --------------------------------------------------

results = []

for key, path in sorted(observations.items()):

    with rasterio.open(path) as src:
        fmask = src.read(1)
        nodata = src.nodata

    total_pixels = fmask.size

    if nodata is None:
        nodata = NODATA_VALUE

    valid_mask = fmask != nodata

    recognized_mask = (
        valid_mask
        & np.isin(fmask, list(lookup_values))
    )

    nodata_count = np.count_nonzero(~valid_mask)
    recognized_count = np.count_nonzero(recognized_mask)

    unknown_count = np.count_nonzero(
        valid_mask & ~recognized_mask
    )

    if recognized_count == 0:
        print(f"WARNING: No recognized pixels in {path.name}")
        continue

    # Count pixels belonging to each lookup category.

    def count_category(values):
        return np.count_nonzero(
            recognized_mask & np.isin(fmask, list(values))
        )

    counts = {
        "cloud": count_category(cloud_values),
        "cloud_shadow": count_category(shadow_values),
        "snow_ice": count_category(snow_values),
        "adjacent_to_cloud": count_category(adjacent_values),
        "water": count_category(water_values),
        "low_aerosol": count_category(aerosol_values["Low aerosol"]),
        "moderate_aerosol": count_category(
            aerosol_values["Moderate aerosol"]
        ),
        "high_aerosol": count_category(aerosol_values["High aerosol"]),
    }

    # A clear pixel has no cloud, shadow, snow or adjacency flag.
    # Water is retained because water is not inherently poor imagery.

    excluded_values = (
        cloud_values
        | shadow_values
        | snow_values
        | adjacent_values
    )

    clear_mask = (
        recognized_mask
        & ~np.isin(fmask, list(excluded_values))
    )

    clear_count = np.count_nonzero(clear_mask)

    # A conservative estimate also excludes high-aerosol pixels.
    conservative_mask = (
        clear_mask
        & ~np.isin(fmask, list(aerosol_values["High aerosol"]))
    )

    conservative_count = np.count_nonzero(conservative_mask)

    # All quality percentages below use recognized pixels
    # as their denominator, unless explicitly stated otherwise.

    metrics = {
        "date_doy": key[0],
        "area_id": key[1],
        "tile": key[2],
        "total_pixels": total_pixels,
        "valid_pixels": np.count_nonzero(valid_mask),
        "recognized_pixels": recognized_count,
        "nodata_pixels": nodata_count,
        "unclassified_pixels": unknown_count,
    }

    for name, count in counts.items():
        metrics[f"{name}_pct"] = (
            count / recognized_count
        ) * 100

    metrics["nodata_pct"] = (
        nodata_count / total_pixels
    ) * 100

    metrics["unclassified_pct"] = (
        unknown_count / total_pixels
    ) * 100

    metrics["clear_pixels_pct"] = (
        clear_count / recognized_count
    ) * 100

    metrics["conservative_usable_pct"] = (
        conservative_count / recognized_count
    ) * 100

    results.append(metrics)

    print(
        f"Processed: {key[0]} | "
        f"Area {key[1]} | Tile {key[2]}"
    )


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

if not results:
    raise RuntimeError("No observations could be analyzed.")

df = pd.DataFrame(results)

df.to_csv(OUTPUT_FILE, index=False)


# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 55)
print("SPECTRAAI FMASK ANALYSIS SUMMARY")
print("=" * 55)

print(f"Observations analyzed: {len(df)}")

summary_columns = [
    "cloud_pct",
    "cloud_shadow_pct",
    "snow_ice_pct",
    "adjacent_to_cloud_pct",
    "water_pct",
    "high_aerosol_pct",
    "clear_pixels_pct",
    "conservative_usable_pct",
]

print("\nAverage percentages across observations:")

for column in summary_columns:
    print(
        f"{column.replace('_pct', '').replace('_', ' ').title():30s}"
        f"{df[column].mean():8.2f}%"
    )

print("\nResults saved to:")
print(OUTPUT_FILE)

print("\nFMASK ANALYSIS COMPLETED SUCCESSFULLY!")
