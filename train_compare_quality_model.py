from pathlib import Path
import re

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit

from image_features import extract_features

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "HLS_S30"
MODEL_DIR = BASE_DIR / "model"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"
PREVIEW_DIR = BASE_DIR / "data" / "previews"
EXISTING_MODEL_PATH = MODEL_DIR / "quality_model.pkl"
CANDIDATE_MODEL_PATH = MODEL_DIR / "quality_model_hls_candidate.pkl"
PREDICTIONS_PATH = OUTPUT_DIR / "ml_hls_candidate_comparison.csv"
MATRIX_PATH = PREVIEW_DIR / "ml_hls_candidate_confusion_matrix.png"
METRICS_PATH = OUTPUT_DIR / "ml_hls_candidate_metrics.txt"

IMAGE_SIZE = 256
VARIANTS_PER_OBSERVATION = 30
RANDOM_SEED = 42
BANDS = ("B02", "B03", "B04")
REQUIRED_LAYERS = (*BANDS, "FMASK")
FILENAME_PATTERN = re.compile(
    r"HLSS30\.020_(B02|B03|B04|B8A|Fmask)_doy(\d{7})_aid(\d{4})_(\d{2}N)\.tif$",
    re.IGNORECASE,
)
EXCLUDED_FMASK_BITS = 2 | 4 | 8 | 16
FEATURE_NAMES = ["brightness", "contrast", "sharpness", "edge_density", "noise_indicator"]
CLASS_ORDER = ["GOOD", "MODERATE", "POOR"]


def read_scaled_band(path, size=IMAGE_SIZE):
    with rasterio.open(path) as src:
        data = src.read(1, out_shape=(size, size), resampling=Resampling.average).astype(np.float32)
        valid = src.read_masks(1, out_shape=(size, size), resampling=Resampling.nearest) > 0
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0
        data = data * (1.0 if scale is None else float(scale)) + (0.0 if offset is None else float(offset))
        valid &= np.isfinite(data)
        data[~valid] = np.nan
        return data, valid


def build_rgb_composite(files):
    bands, masks = {}, []
    for band in BANDS:
        bands[band], valid = read_scaled_band(files[band])
        masks.append(valid)

    with rasterio.open(files["FMASK"]) as src:
        fmask = src.read(
            1, out_shape=(IMAGE_SIZE, IMAGE_SIZE), resampling=Resampling.nearest
        ).astype(np.uint8)
        fmask_valid = src.read_masks(
            1, out_shape=(IMAGE_SIZE, IMAGE_SIZE), resampling=Resampling.nearest
        ) > 0

    common_valid = np.logical_and.reduce(masks) & fmask_valid & (fmask != 255)
    clear = common_valid & ((fmask & EXCLUDED_FMASK_BITS) == 0)
    sample_mask = clear if np.count_nonzero(clear) >= 100 else common_valid
    if np.count_nonzero(sample_mask) < 100:
        raise ValueError("Too few valid pixels to build a stable RGB composite")

    channels = []
    for channel in [bands["B04"], bands["B03"], bands["B02"]]:
        low, high = np.nanpercentile(channel[sample_mask], [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Invalid reflectance stretch for an RGB channel")
        stretched = np.clip((channel - low) / (high - low), 0, 1)
        stretched = np.nan_to_num(stretched, nan=0.5, posinf=0.5, neginf=0.5)
        stretched[~sample_mask] = float(np.nanmedian(stretched[sample_mask]))
        channels.append((stretched * 255).astype(np.uint8))

    return np.stack(channels, axis=-1)


def degrade_image(image, rng):
    result = image.copy()
    severity = 0.0

    blur_draw = rng.random()
    kernel_size = 3 if blur_draw < 0.35 else 5 if blur_draw < 0.70 else 9
    if kernel_size > 3:
        result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        severity += {5: 0.20, 9: 0.40}[kernel_size]

    contrast_factor = rng.uniform(0.35, 1.15)
    result = np.clip(
        (result.astype(np.float32) - 127.5) * contrast_factor + 127.5, 0, 255
    ).astype(np.uint8)
    severity += max(0.0, 0.85 - contrast_factor) * 0.65

    brightness_shift = rng.uniform(-65, 65)
    result = np.clip(
        result.astype(np.float32) + brightness_shift, 0, 255
    ).astype(np.uint8)
    severity += max(0.0, abs(brightness_shift) - 25) / 100

    noise_level = rng.uniform(0, 24)
    noise = rng.normal(0, noise_level, result.shape)
    result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    severity += noise_level / 40

    label = "GOOD" if severity < 0.30 else "MODERATE" if severity < 0.65 else "POOR"
    return result, label


def discover_observations():
    observations = {}
    for path in DATA_DIR.glob("*.tif"):
        match = FILENAME_PATTERN.match(path.name)
        if match:
            layer, date, area_id, tile = match.groups()
            observations.setdefault((date, area_id, tile), {})[layer.upper()] = path

    return {
        key: files
        for key, files in observations.items()
        if all(layer in files for layer in REQUIRED_LAYERS)
    }


def build_samples(observations):
    rng = np.random.default_rng(RANDOM_SEED)
    rows, skipped = [], []

    for index, (key, files) in enumerate(sorted(observations.items()), start=1):
        observation_id = "_".join(key)
        # Acquisition identity intentionally excludes tile. All tiles from the
        # same date + area acquisition must remain in one split.
        acquisition_id = "_".join(key[:2])

        try:
            image = build_rgb_composite(files)
            features = extract_features(image, color_order="BGR")
            rows.append({
                "observation_id": observation_id,
                "acquisition_id": acquisition_id,
                "sample_index": 0,
                "synthetic_reference_label": "GOOD",
                **{name: float(features[name]) for name in FEATURE_NAMES},
            })

            for sample_index in range(1, VARIANTS_PER_OBSERVATION + 1):
                degraded, label = degrade_image(image, rng)
                features = extract_features(degraded, color_order="BGR")
                rows.append({
                    "observation_id": observation_id,
                    "acquisition_id": acquisition_id,
                    "sample_index": sample_index,
                    "synthetic_reference_label": label,
                    **{name: float(features[name]) for name in FEATURE_NAMES},
                })

            print(f"  {index}/{len(observations)} processed: {observation_id}")
        except Exception as exc:
            skipped.append((observation_id, str(exc)))
            print(f"  SKIPPED {observation_id}: {exc}")

    return pd.DataFrame(rows), skipped


def load_original_model():
    saved = joblib.load(EXISTING_MODEL_PATH)
    model = saved["model"] if isinstance(saved, dict) and "model" in saved else saved

    if isinstance(saved, dict):
        saved_features = saved.get("feature_names", FEATURE_NAMES)
        if list(saved_features) != FEATURE_NAMES:
            raise ValueError(
                "Original model feature order mismatch. "
                f"Found {saved_features}; expected {FEATURE_NAMES}"
            )

    return model


def print_metrics(title, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{title}\nAccuracy: {accuracy:.2%}")
    print(classification_report(
        y_true, y_pred, labels=CLASS_ORDER, zero_division=0
    ))
    return accuracy


def main():
    for folder in (MODEL_DIR, OUTPUT_DIR, PREVIEW_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"HLS dataset folder not found: {DATA_DIR}")
    if not EXISTING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Existing model not found: {EXISTING_MODEL_PATH}")

    observations = discover_observations()
    if len(observations) < 8:
        raise RuntimeError(
            f"Found only {len(observations)} complete tile observations; "
            "at least 8 are required."
        )

    print(f"Complete HLS tile observations discovered: {len(observations)}")
    print("Generating the synthetic benchmark...")
    dataset, skipped = build_samples(observations)

    if dataset.empty:
        raise RuntimeError("No HLS observations could be processed.")

    acquisition_count = dataset["acquisition_id"].nunique()
    if acquisition_count < 4:
        raise RuntimeError(
            f"Only {acquisition_count} unique acquisitions were processed; "
            "at least 4 are required for a grouped holdout."
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_SEED
    )
    train_idx, test_idx = next(splitter.split(
        dataset,
        dataset["synthetic_reference_label"],
        groups=dataset["acquisition_id"],
    ))

    train = dataset.iloc[train_idx].copy()
    test = dataset.iloc[test_idx].copy()
    train_acquisitions = set(train["acquisition_id"])
    test_acquisitions = set(test["acquisition_id"])
    shared_acquisitions = train_acquisitions.intersection(test_acquisitions)

    if shared_acquisitions:
        raise RuntimeError(
            "ACQUISITION LEAKAGE DETECTED: "
            f"{sorted(shared_acquisitions)}"
        )
    if set(train["synthetic_reference_label"]) != set(CLASS_ORDER):
        raise RuntimeError("Training split does not contain all three classes.")

    print("\n=== ACQUISITION-LEVEL GROUPED TRAIN / TEST SPLIT ===")
    print(f"Training acquisitions: {len(train_acquisitions)}")
    print(f"Testing acquisitions: {len(test_acquisitions)}")
    print(f"Shared acquisitions: {sorted(shared_acquisitions)}")
    print(f"Acquisition leakage: {bool(shared_acquisitions)}")
    print(f"Training tile observations: {train['observation_id'].nunique()}")
    print(f"Testing tile observations: {test['observation_id'].nunique()}")
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")

    candidate = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    candidate.fit(train[FEATURE_NAMES], train["synthetic_reference_label"])
    candidate_predictions = candidate.predict(test[FEATURE_NAMES])
    candidate_accuracy = print_metrics(
        "=== HLS-TRAINED CANDIDATE ON HELD-OUT ACQUISITIONS ===",
        test["synthetic_reference_label"],
        candidate_predictions,
    )

    # Evaluate the original model on exactly the same held-out rows. This avoids
    # relying on predictions from the previous, tile-level split.
    original_model = load_original_model()
    original_predictions = original_model.predict(test[FEATURE_NAMES])
    baseline_accuracy = print_metrics(
        "=== ORIGINAL MODEL ON THE SAME HELD-OUT ACQUISITIONS ===",
        test["synthetic_reference_label"],
        original_predictions,
    )

    comparison = test.copy()
    comparison["candidate_prediction"] = candidate_predictions
    comparison["original_model_prediction"] = original_predictions
    comparison.to_csv(PREDICTIONS_PATH, index=False)

    joblib.dump({
        "model": candidate,
        "feature_names": FEATURE_NAMES,
        "classes": CLASS_ORDER,
        "training_acquisitions": sorted(train_acquisitions),
        "held_out_acquisitions": sorted(test_acquisitions),
        "training_observations": sorted(train["observation_id"].unique()),
        "held_out_observations": sorted(test["observation_id"].unique()),
        "split_group": "acquisition_id (date + area_id; tile excluded)",
        "training_note": (
            "Trained on synthetic degradations generated from HLS RGB composites. "
            "Labels represent controlled synthetic degradation severity, not "
            "independently verified real-world quality."
        ),
    }, CANDIDATE_MODEL_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_predictions(
        test["synthetic_reference_label"],
        candidate_predictions,
        labels=CLASS_ORDER,
        display_labels=CLASS_ORDER,
        cmap="Blues",
        colorbar=False,
        ax=axes[0],
    )
    axes[0].set_title("HLS-trained candidate")
    ConfusionMatrixDisplay.from_predictions(
        test["synthetic_reference_label"],
        original_predictions,
        labels=CLASS_ORDER,
        display_labels=CLASS_ORDER,
        cmap="Oranges",
        colorbar=False,
        ax=axes[1],
    )
    axes[1].set_title("Original model baseline")
    fig.suptitle("Acquisition-held-out synthetic quality benchmark")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = [
        "SPECTRAAI — HLS QUALITY MODEL EXPERIMENT",
        "========================================",
        f"Complete tile observations discovered: {len(observations)}",
        f"Successfully processed tile observations: {dataset['observation_id'].nunique()}",
        f"Unique acquisitions: {acquisition_count}",
        f"Skipped tile observations: {len(skipped)}",
        "Split grouping: acquisition_id (date + area_id; tile excluded)",
        f"Training acquisitions: {len(train_acquisitions)}",
        f"Testing acquisitions: {len(test_acquisitions)}",
        f"Shared acquisitions: {len(shared_acquisitions)}",
        f"Training samples: {len(train)}",
        f"Testing samples: {len(test)}",
        f"Candidate accuracy: {candidate_accuracy:.4f}",
        f"Original baseline accuracy on same test rows: {baseline_accuracy:.4f}",
        "",
        "LIMITATION: All reference labels are derived from controlled synthetic",
        "degradation severity. This is not validation against independently",
        "labelled real-world image quality.",
    ]
    METRICS_PATH.write_text("\n".join(summary), encoding="utf-8")

    print("\n=== EXPERIMENT COMPLETE ===")
    print(f"Candidate model saved separately: {CANDIDATE_MODEL_PATH}")
    print(f"Comparison predictions saved: {PREDICTIONS_PATH}")
    print(f"Confusion matrix saved: {MATRIX_PATH}")
    print(f"Experiment summary saved: {METRICS_PATH}")
    print("The existing quality_model.pkl has NOT been changed.")


if __name__ == "__main__":
    main()
