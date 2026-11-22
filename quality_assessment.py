
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "metadata" / "multispectral_features.csv"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"
OUTPUT_FILE = OUTPUT_DIR / "quality_assessment.csv"

CHART_DIR = BASE_DIR / "data" / "previews"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def assign_category(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 50:
        return "Fair"
    return "Poor"


def main():

    print("\nSPECTRAAI — QUALITY ASSESSMENT ENGINE")
    print("=" * 50)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(INPUT_FILE)

    required = [
        "date_doy",
        "area_id",
        "tile",
        "clear_reflectance_coverage_pct",
        "clear_pixels_pct",
        "high_aerosol_pct",
        "cloud_pct",
        "cloud_shadow_pct",
        "snow_ice_pct",
        "red_laplacian_variance"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Primary score: percentage of the complete raster
    # that contains valid reflectance and passes our
    # conservative Fmask clear-pixel definition.
    df["quality_score"] = (
        df["clear_reflectance_coverage_pct"].clip(0, 100)
    )

    df["quality_category"] = df["quality_score"].apply(
        assign_category
    )

    # Aerosol is a separate warning, not an automatic
    # deduction from the quality score.
    df["aerosol_warning"] = df["high_aerosol_pct"].apply(
        lambda x: "High aerosol flag prevalence"
        if x >= 50 else "No high aerosol prevalence warning"
    )

    df["quality_basis"] = (
        "Valid reflectance coverage after Fmask cloud, "
        "shadow, snow and adjacency exclusion"
    )

    df = df.sort_values(
        "quality_score",
        ascending=False
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        float_format="%.6f"
    )

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------

    print("\nOBSERVATIONS:", len(df))

    print("\nQUALITY CATEGORY DISTRIBUTION:")

    print(
        df["quality_category"]
        .value_counts()
        .reindex(
            ["Excellent", "Good", "Fair", "Poor"],
            fill_value=0
        )
        .to_string()
    )

    print("\nQUALITY SCORE STATISTICS:")

    print(
        df["quality_score"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nTOP 5 OBSERVATIONS:")

    print(
        df[
            [
                "date_doy",
                "area_id",
                "tile",
                "quality_score",
                "quality_category"
            ]
        ].head(5).to_string(index=False)
    )

    print("\nBOTTOM 5 OBSERVATIONS:")

    print(
        df[
            [
                "date_doy",
                "area_id",
                "tile",
                "quality_score",
                "quality_category"
            ]
        ].tail(5).to_string(index=False)
    )

    # ------------------------------------------------
    # CHART 1: CATEGORY DISTRIBUTION
    # ------------------------------------------------

    category_order = ["Excellent", "Good", "Fair", "Poor"]

    counts = (
        df["quality_category"]
        .value_counts()
        .reindex(category_order, fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    counts.plot(kind="bar")

    plt.title("SpectraAI Quality Category Distribution")
    plt.xlabel("Quality Category")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "quality_category_distribution.png",
        dpi=300
    )

    plt.close()

    # ------------------------------------------------
    # CHART 2: OBSERVATION SCORES
    # ------------------------------------------------

    labels = (
        df["date_doy"].astype(str)
        + " | "
        + df["area_id"].astype(str)
        + " | "
        + df["tile"].astype(str)
    )

    plt.figure(figsize=(13, 7))
    plt.barh(labels, df["quality_score"])

    plt.axvline(90, linestyle="--", label="Excellent threshold")
    plt.axvline(75, linestyle="--", label="Good threshold")
    plt.axvline(50, linestyle="--", label="Fair threshold")

    plt.xlabel("Quality Score")
    plt.ylabel("Satellite Observation")
    plt.title("SpectraAI Observation Quality Scores")
    plt.xlim(0, 105)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "observation_quality_ranking.png",
        dpi=300
    )

    plt.close()

    print("\nOUTPUT FILES:")
    print(OUTPUT_FILE)
    print(CHART_DIR / "quality_category_distribution.png")
    print(CHART_DIR / "observation_quality_ranking.png")

    print("\nQUALITY ASSESSMENT COMPLETED.")


if __name__ == "__main__":
    main()
