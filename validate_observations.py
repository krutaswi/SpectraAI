
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/HLS_S30")
OUTPUT_DIR = Path("data/metadata")

EXPECTED_LAYERS = {"B02", "B03", "B04", "B8A", "FMASK"}

df = pd.read_csv(OUTPUT_DIR / "dataset_inventory.csv")

# Each unique date, area and tile represents one observation
group_columns = ["date", "area_id", "tile"]

observations = []

for key, group in df.groupby(group_columns):

    present = set(group["layer"].str.upper())
    missing = EXPECTED_LAYERS - present

    observations.append({
        "date": key[0],
        "area_id": key[1],
        "tile": key[2],
        "layer_count": len(present),
        "missing_layers": ", ".join(sorted(missing)),
        "complete": len(missing) == 0
    })

result = pd.DataFrame(observations)

result.to_csv(
    OUTPUT_DIR / "observation_validation.csv",
    index=False
)

print("\nSPECTRAAI OBSERVATION VALIDATION")
print("-" * 40)

print("Total observations:", len(result))
print("Complete observations:", result["complete"].sum())
print("Incomplete observations:", (~result["complete"]).sum())

if result["complete"].all():
    print("\nSUCCESS! Every observation has all five layers.")
else:
    print("\nSome observations have missing layers:")
    print(result.loc[~result["complete"]].to_string(index=False))

print("\nValidation report saved to:")
print(OUTPUT_DIR / "observation_validation.csv")
