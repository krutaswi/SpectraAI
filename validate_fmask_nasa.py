
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

RASTER_DIR = ROOT / "data" / "raw" / "HLS_S30"
METADATA_DIR = ROOT / "data" / "metadata"

NASA_CSV = METADATA_DIR / "HLSS30-020-Fmask-Statistics-QA.csv"

OUTPUT_CSV = METADATA_DIR / "fmask_nasa_validation.csv"


# --------------------------------------------------
# LOAD NASA STATISTICS
# --------------------------------------------------

nasa = pd.read_csv(NASA_CSV)

# Convert date to the same year + day-of-year format
nasa["date_doy"] = pd.to_datetime(
    nasa["Date"]
).dt.strftime("%Y%j")

nasa["area_id"] = (
    nasa["aid"].astype(str).str.replace("aid", "", regex=False)
)

nasa["tile"] = nasa["UTM Zone"].astype(str)

# Fmask pixel-value columns
value_columns = [
    column for column in nasa.columns
    if str(column).isdigit()
]

for column in value_columns:
    nasa[column] = pd.to_numeric(
        nasa[column], errors="coerce"
    ).fillna(0)


# --------------------------------------------------
# FIND RASTER FILES
# --------------------------------------------------

pattern = re.compile(
    r"HLSS30\.020_Fmask_"
    r"doy(\d{7})_aid(\d+)_([A-Za-z0-9]+)\.tif$",
    re.IGNORECASE
)

raster_files = {}

for path in RASTER_DIR.glob("*.tif"):
    match = pattern.match(path.name)

    if not match:
        continue

    date, area, tile = match.groups()

    key = (date, area, tile)
    raster_files[key] = path


# --------------------------------------------------
# COMPARE EACH OBSERVATION
# --------------------------------------------------

validation_results = []

for _, row in nasa.iterrows():

    key = (
        row["date_doy"],
        row["area_id"],
        row["tile"]
    )

    if key not in raster_files:
        validation_results.append({
            "date_doy": key[0],
            "area_id": key[1],
            "tile": key[2],
            "status": "RASTER_NOT_FOUND",
            "mismatched_values": None,
            "total_absolute_difference": None
        })
        continue

    path = raster_files[key]

    with rasterio.open(path) as src:
        raster = src.read(1)
        nodata = src.nodata

    raster_values, raster_counts = np.unique(
        raster, return_counts=True
    )

    raster_distribution = {
        str(int(value)): int(count)
        for value, count in zip(raster_values, raster_counts)
        if nodata is None or value != nodata
    }

    mismatches = []
    total_difference = 0

    for column in value_columns:

        nasa_count = int(row[column])
        raster_count = raster_distribution.get(column, 0)

        difference = abs(nasa_count - raster_count)

        if difference != 0:
            mismatches.append(
                f"{column}: NASA={nasa_count}, Raster={raster_count}"
            )

        total_difference += difference

    validation_results.append({
        "date_doy": key[0],
        "area_id": key[1],
        "tile": key[2],
        "status": (
            "MATCH"
            if not mismatches
            else "MISMATCH"
        ),
        "mismatched_values": " | ".join(mismatches),
        "total_absolute_difference": total_difference
    })


# --------------------------------------------------
# SAVE AND REPORT
# --------------------------------------------------

result = pd.DataFrame(validation_results)

result.to_csv(OUTPUT_CSV, index=False)

print("\nSPECTRAAI — NASA FMASK CROSS-VALIDATION")
print("=" * 55)

print("Total NASA QA records:", len(nasa))
print("Total raster files:", len(raster_files))
print("Validated records:", len(result))

print("\nValidation status:")
print(result["status"].value_counts().to_string())

print("\nMismatched records:")

mismatched = result[result["status"] != "MATCH"]

if mismatched.empty:
    print("None. All compared pixel-value counts match.")
else:
    print(mismatched.to_string(index=False))

print("\nSaved report:")
print(OUTPUT_CSV)
