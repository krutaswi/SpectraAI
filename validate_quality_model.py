from pathlib import Path
import re

import cv2
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

from image_features import extract_features


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw" / "HLS_S30"
MODEL_PATH = BASE_DIR / "model" / "quality_model.pkl"
OUTPUT_DIR = BASE_DIR / "data" / "metadata"
PREVIEW_DIR = BASE_DIR / "data" / "previews"
REPORT_PATH = OUTPUT_DIR / "ml_cross_observation_validation.csv"
MATRIX_PATH = PREVIEW_DIR / "ml_cross_observation_confusion_matrix.png"

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
FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "noise_indicator",
]
CLASS_ORDER = ["GOOD", "MODERATE", "POOR"]


def read_scaled_band(path, size=IMAGE_SIZE):
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(size, size),
            resampling=Resampling.average,
        ).astype(np.float32)
        valid = src.read_masks(
            1,
            out_shape=(size, size),
            resampling=Resampling.nearest,
        ) > 0
        scale = src.scales[0] if src.scales else 1.0
        offset = src.offsets[0] if src.offsets else 0.0
        scale = 1.0 if scale is None else float(scale)
        offset = 0.0 if offset is None else float(offset)
        data = data * scale + offset
        valid &= np.isfinite(data)
        data[~valid] = np.nan
        return data, valid


def build_rgb_composite(files):
    bands = {}
    masks = []
    for band in BANDS:
        bands[band], valid = read_scaled_band(files[band])
        masks.append(valid)

    with rasterio.open(files["FMASK"]) as src:
        fmask = src.read(
            1,
            out_shape=(IMAGE_SIZE, IMAGE_SIZE),
            resampling=Resampling.nearest,
        ).astype(np.uint8)
        fmask_valid = src.read_masks(
            1,
            out_shape=(IMAGE_SIZE, IMAGE_SIZE),
            resampling=Resampling.nearest,
        ) > 0

    common_valid = np.logical_and.reduce(masks) & fmask_valid & (fmask != 255)
    clear = common_valid & ((fmask & EXCLUDED_FMASK_BITS) == 0)

    # Prefer Fmask-clear pixels. Fall back to all valid pixels only when
    # the clear sample is too small to estimate stable display percentiles.
    sample_mask = clear if np.count_nonzero(clear) >= 100 else common_valid
    if np.count_nonzero(sample_mask) < 100:
        raise ValueError("Too few valid pixels to build a stable RGB composite")

    # OpenCV expects BGR. Use Red, Green, Blue channel order below.
    channel_arrays = [bands["B04"], bands["B03"], bands["B02"]]
    channels = []
    for channel in channel_arrays:
        values = channel[sample_mask]
        low, high = np.nanpercentile(values, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            raise ValueError("Invalid reflectance stretch for an RGB channel")
        stretched = np.clip((channel - low) / (high - low), 0, 1)
        stretched = np.nan_to_num(stretched, nan=0.5, posinf=0.5, neginf=0.5)
        # Fill masked pixels with the median display value instead of black,
        # which would otherwise introduce artificial edges into the features.
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
        (result.astype(np.float32) - 127.5) * contrast_factor + 127.5,
        0,
        255,
    ).astype(np.uint8)
    severity += max(0.0, 0.85 - contrast_factor) * 0.65

    brightness_shift = rng.uniform(-65, 65)
    result = np.clip(result.astype(np.float32) + brightness_shift, 0, 255).astype(np.uint8)
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
        if not match:
            continue
        layer, date, area_id, tile = match.groups()
        key = (date, area_id, tile)
        observations.setdefault(key, {})[layer.upper()] = path

    complete = {
        key: files
        for key, files in observations.items()
        if all(layer in files for layer in REQUIRED_LAYERS)
    }
    return complete


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"HLS dataset folder not found: {DATA_DIR}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    saved = joblib.load(MODEL_PATH)
    model = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    model_features = saved.get("feature_names", FEATURE_NAMES) if isinstance(saved, dict) else FEATURE_NAMES
    if list(model_features) != FEATURE_NAMES:
        raise ValueError(
            f"Saved model feature order does not match this validator. "
            f"Model: {model_features}; expected: {FEATURE_NAMES}"
        )

    observations = discover_observations()
    if len(observations) < 8:
        raise RuntimeError(
            f"Found only {len(observations)} complete observations. "
            "At least 8 are required for a meaningful grouped holdout."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    skipped = []
    print(f"Complete HLS observations discovered: {len(observations)}")
    print("Building RGB composites and synthetic degradation samples...")

    for index, (key, files) in enumerate(sorted(observations.items()), start=1):
        observation_id = "_".join(key)
        try:
            base_image = build_rgb_composite(files)
            samples = [(base_image, "GOOD")]
            for _ in range(VARIANTS_PER_OBSERVATION):
                variant, label = degrade_image(base_image, rng)
                samples.append((variant, label))

            for sample_index, (image, label) in enumerate(samples):
                features = extract_features(image, color_order="BGR")
                rows.append({
                    "observation_id": observation_id,
                    "sample_index": sample_index,
                    "synthetic_reference_label": label,
                    **{name: features[name] for name in FEATURE_NAMES},
                })
            print(f"  {index}/{len(observations)} processed: {observation_id}")
        except Exception as exc:
            skipped.append((observation_id, str(exc)))
            print(f"  SKIPPED {observation_id}: {exc}")

    if len({row["observation_id"] for row in rows}) < 8:
        raise RuntimeError("Fewer than 8 observations could be processed successfully.")

    data = pd.DataFrame(rows)
    X = data[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = data["synthetic_reference_label"].to_numpy()
    groups = data["observation_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    # Critical leakage guard: no source observation may occur in both sets.
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Observation leakage detected: {sorted(overlap)}")

    predictions = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx]) if hasattr(model, "predict_proba") else None

    data.loc[test_idx, "model_prediction"] = predictions
    if probabilities is not None:
        for class_index, class_name in enumerate(model.classes_):
            data.loc[test_idx, f"probability_{class_name.lower()}"] = probabilities[:, class_index]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    data.loc[test_idx].to_csv(REPORT_PATH, index=False, float_format="%.6f")

    accuracy = accuracy_score(y[test_idx], predictions)
    print("\n=== CROSS-OBSERVATION SYNTHETIC VALIDATION ===")
    print(f"Usable source observations: {len(train_groups) + len(test_groups)}")
    print(f"Other source observations (outside the held-out set): {len(train_groups)}")
    print(f"Held-out source observations: {len(test_groups)}")
    print(f"Held-out samples: {len(test_idx)}")
    print(f"Source-observation overlap: {len(overlap)}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y[test_idx], predictions, labels=CLASS_ORDER, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y[test_idx], predictions, labels=CLASS_ORDER,
        display_labels=CLASS_ORDER, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title("SpectraAI — Held-out Observation Validation\nSynthetic degradation labels")
    fig.tight_layout()
    fig.savefig(MATRIX_PATH, dpi=200)
    plt.close(fig)

    print(f"\nHeld-out predictions saved: {REPORT_PATH}")
    print(f"Confusion matrix saved: {MATRIX_PATH}")
    if skipped:
        print(f"\nSkipped observations: {len(skipped)}")
        for observation_id, reason in skipped:
            print(f"  {observation_id}: {reason}")

    print("\nIMPORTANT: This evaluates the existing model on synthetic degradations of held-out HLS scenes.")
    print("It is not validation against independently labelled real-world image-quality outcomes.")


if __name__ == "__main__":
    main()
