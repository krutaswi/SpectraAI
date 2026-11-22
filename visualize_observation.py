
from pathlib import Path
import re

import numpy as np
import rasterio
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "raw" / "HLS_S30"
PREVIEW_DIR = PROJECT_DIR / "data" / "previews"

PREVIEW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIND AND GROUP HLS FILES
# --------------------------------------------------

pattern = re.compile(
    r"HLSS30\.020_(B02|B03|B04|B8A|Fmask)_"
    r"doy(\d{7})_aid(\d+)_([A-Za-z0-9]+)\.tif$",
    re.IGNORECASE
)

observations = {}

for file_path in DATA_DIR.glob("*.tif"):
    match = pattern.match(file_path.name)

    if not match:
        continue

    band, date, area_id, tile = match.groups()

    key = (date, area_id, tile)
    observations.setdefault(key, {})
    observations[key][band.upper()] = file_path


required_layers = {"B02", "B03", "B04", "B8A", "FMASK"}

complete = {
    key: files
    for key, files in observations.items()
    if required_layers.issubset(files.keys())
}

if not complete:
    raise RuntimeError(
        "No complete observations found. Check the TIFF filenames and folder."
    )


# --------------------------------------------------
# SELECT FIRST COMPLETE OBSERVATION
# --------------------------------------------------

selected_key = sorted(complete.keys())[0]
selected_files = complete[selected_key]

print("\nSPECTRAAI — OBSERVATION INSPECTION")
print("-" * 45)
print(f"Date (year + day of year): {selected_key[0]}")
print(f"Area ID: {selected_key[1]}")
print(f"Tile: {selected_key[2]}")


# --------------------------------------------------
# READ RASTER AND INSPECT METADATA
# --------------------------------------------------

def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)

        metadata = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": src.transform,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": src.bounds,
            "tags": src.tags()
        }

    return data, metadata


raster_data = {}
raster_metadata = {}

for layer, path in selected_files.items():
    data, metadata = read_raster(path)
    raster_data[layer] = data
    raster_metadata[layer] = metadata

    print(f"\n{layer}")
    print(f"  Dimensions: {metadata['width']} x {metadata['height']}")
    print(f"  CRS: {metadata['crs']}")
    print(f"  Data type: {metadata['dtype']}")
    print(f"  NoData value: {metadata['nodata']}")
    print(f"  Raw minimum: {np.nanmin(data)}")
    print(f"  Raw maximum: {np.nanmax(data)}")
    print(f"  Tags: {metadata['tags']}")


# --------------------------------------------------
# PREPARE TRUE-COLOR RGB PREVIEW
# --------------------------------------------------

def percentile_stretch(band):
    valid = np.isfinite(band)

    if not np.any(valid):
        return np.zeros_like(band, dtype=np.float32)

    low, high = np.percentile(band[valid], [2, 98])

    if high <= low:
        return np.zeros_like(band, dtype=np.float32)

    stretched = (band - low) / (high - low)
    return np.clip(stretched, 0, 1)


# HLS true color: Red = B04, Green = B03, Blue = B02
red = raster_data["B04"]
green = raster_data["B03"]
blue = raster_data["B02"]

# Exclude declared NoData pixels from the RGB preview.
valid = np.ones(red.shape, dtype=bool)

for layer in ["B02", "B03", "B04"]:
    nodata = raster_metadata[layer]["nodata"]

    if nodata is not None:
        valid &= raster_data[layer] != nodata

valid &= np.isfinite(red) & np.isfinite(green) & np.isfinite(blue)

rgb = np.stack([
    percentile_stretch(np.where(valid, red, np.nan)),
    percentile_stretch(np.where(valid, green, np.nan)),
    percentile_stretch(np.where(valid, blue, np.nan))
], axis=-1)

rgb[~valid] = 0


# --------------------------------------------------
# DISPLAY AND SAVE PREVIEW
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].imshow(rgb)
axes[0].set_title("HLS True-Color Satellite Image")
axes[0].axis("off")

fmask = raster_data["FMASK"]

image = axes[1].imshow(fmask, cmap="tab20")
axes[1].set_title("Fmask — Raw Pixel Values")
axes[1].axis("off")

fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

fig.suptitle(
    f"SpectraAI | Date: {selected_key[0]} | Area: {selected_key[1]}",
    fontsize=13
)

plt.tight_layout()

output_path = PREVIEW_DIR / "first_observation_preview.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()

print("\n" + "-" * 45)
print("PREVIEW GENERATED SUCCESSFULLY!")
print(f"Saved at: {output_path}")
