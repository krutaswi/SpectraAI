
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "HLS_S30"
METADATA_DIR = PROJECT_DIR / "data" / "metadata"

LOOKUP_FILE = METADATA_DIR / "HLSS30-020-Fmask-lookup.csv"

lookup = pd.read_csv(LOOKUP_FILE)
lookup["Value"] = lookup["Value"].astype(int)

pattern = re.compile(
    r"HLSS30\.020_Fmask_"
    r"doy(\d{7})_aid(\d+)_([A-Za-z0-9]+)\.tif$",
    re.IGNORECASE
)

files = []

for path in DATA_DIR.glob("*.tif"):
    match = pattern.match(path.name)
    if match:
        date, area, tile = match.groups()
        files.append((date, area, tile, path))

files.sort()

if not files:
    raise RuntimeError("No Fmask files found.")

# Inspect the first observation.
date, area, tile, path = files[0]

with rasterio.open(path) as src:
    data = src.read(1)
    nodata = src.nodata

print("\nSPECTRAAI — FMASK PIXEL DISTRIBUTION")
print("=" * 55)
print(f"Observation: {date} | Area {area} | Tile {tile}")
print(f"File: {path.name}")
print(f"NoData: {nodata}")

values, counts = np.unique(data, return_counts=True)

distribution = pd.DataFrame({
    "Value": values,
    "Pixel_Count": counts
})

distribution["Percentage"] = (
    distribution["Pixel_Count"] / data.size * 100
).round(3)

distribution = distribution.merge(
    lookup,
    on="Value",
    how="left"
)

print("\nPixel-value distribution for first observation:\n")
print(distribution.to_string(index=False))

# Aggregate pixel counts across every observation.
all_counts = {}

for date, area, tile, path in files:
    with rasterio.open(path) as src:
        data = src.read(1)

    values, counts = np.unique(data, return_counts=True)

    for value, count in zip(values, counts):
        all_counts[int(value)] = (
            all_counts.get(int(value), 0) + int(count)
        )

total_pixels = sum(all_counts.values())

aggregate = pd.DataFrame([
    {
        "Value": value,
        "Pixel_Count": count,
        "Percentage": round(count / total_pixels * 100, 3)
    }
    for value, count in sorted(all_counts.items())
])

aggregate = aggregate.merge(
    lookup,
    on="Value",
    how="left"
)

print("\n" + "=" * 55)
print("AGGREGATED DISTRIBUTION — ALL OBSERVATIONS")
print("=" * 55)
print(aggregate.to_string(index=False))

print("\nTotal observations:", len(files))
print("Total pixels:", total_pixels)
print("\nInspection completed.")
