
from pathlib import Path
import pandas as pd
import re

# Project directories
DATA_DIR = Path("data/raw/HLS_S30")
OUTPUT_DIR = Path("data/metadata")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Identify available satellite layers
layers = ["B02", "B03", "B04", "B8A", "Fmask"]

records = []

# Scan all TIFF files
for file in DATA_DIR.glob("*"):

    if file.suffix.lower() not in [".tif", ".tiff"]:
        continue

    name = file.name

    match = re.search(
        r"_(B02|B03|B04|B8A|Fmask)_doy(\d{7})_aid(\d+)_([0-9]+N)",
        name,
        re.IGNORECASE
    )

    if not match:
        print(f"Unrecognized filename: {name}")
        continue

    layer, date, area_id, tile = match.groups()

    records.append({
        "filename": name,
        "layer": layer.upper(),
        "date": date,
        "area_id": area_id,
        "tile": tile,
        "size_kb": round(file.stat().st_size / 1024, 2)
    })

# Create inventory
df = pd.DataFrame(records)

if df.empty:
    print("No matching TIFF files found.")
else:
    inventory_path = OUTPUT_DIR / "dataset_inventory.csv"

    df.to_csv(inventory_path, index=False)

    print("\nSPECTRAAI DATASET INVENTORY")
    print("-" * 40)

    print(f"Total TIFF files: {len(df)}")

    print("\nFiles per layer:")
    print(df["layer"].value_counts().to_string())

    print("\nUnique acquisition dates:")
    print(df["date"].nunique())

    print("\nUnique area IDs:")
    print(df["area_id"].nunique())

    print("\nInventory saved to:")
    print(inventory_path)
