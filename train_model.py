
from pathlib import Path

import cv2
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from image_features import extract_features


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test.jpg"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "quality_model.pkl"

FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "edge_density",
    "noise_indicator",
]


def degrade_image(image, rng):
    """Create a controlled image variation with a synthetic label."""

    result = image.copy()
    severity = 0.0

    # Blur: simulate loss of fine detail
    blur_level = rng.random()
    if blur_level < 0.35:
        kernel_size = 3
    elif blur_level < 0.70:
        kernel_size = 5
    else:
        kernel_size = 9

    if kernel_size > 3:
        result = cv2.GaussianBlur(
            result, (kernel_size, kernel_size), 0
        )
        severity += {5: 0.20, 9: 0.40}[kernel_size]

    # Contrast degradation
    contrast_factor = rng.uniform(0.35, 1.15)
    result = np.clip(
        (result.astype(np.float32) - 127.5)
        * contrast_factor + 127.5,
        0, 255
    ).astype(np.uint8)

    severity += max(0.0, 0.85 - contrast_factor) * 0.65

    # Brightness variation
    brightness_shift = rng.uniform(-65, 65)
    result = np.clip(
        result.astype(np.float32) + brightness_shift,
        0, 255
    ).astype(np.uint8)

    severity += max(0.0, abs(brightness_shift) - 25) / 100

    # Sensor-like random noise
    noise_level = rng.uniform(0, 24)
    noise = rng.normal(
        0, noise_level, result.shape
    )
    result = np.clip(
        result.astype(np.float32) + noise,
        0, 255
    ).astype(np.uint8)

    severity += noise_level / 40

    # Synthetic training label based on applied degradation
    if severity < 0.30:
        label = "GOOD"
    elif severity < 0.65:
        label = "MODERATE"
    else:
        label = "POOR"

    return result, label


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            "test.jpg not found. Put your test image "
            "inside the SpectraAI project folder."
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError(
            "Could not read test.jpg. Check that it is a valid image."
        )

    # Resize large images for faster training
    image = cv2.resize(image, (256, 256))

    rng = np.random.default_rng(42)
    features_list = []
    labels = []

    print("Generating synthetic training examples...")

    # Include the original image and many degraded variants
    for index in range(800):
        if index == 0:
            variant = image.copy()
            label = "GOOD"
        else:
            variant, label = degrade_image(image, rng)

        features = extract_features(variant)
        features_list.append(
            [features[name] for name in FEATURE_NAMES]
        )
        labels.append(label)

    X = np.asarray(features_list, dtype=np.float64)
    y = np.asarray(labels)

    print("\nSynthetic class distribution:")
    for label in ["GOOD", "MODERATE", "POOR"]:
        print(f"{label}: {np.sum(y == label)}")

    # Split examples into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n=== SpectraAI Model Evaluation ===")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "classes": list(model.classes_),
            "training_type": "synthetic_degradation",
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Next: connect this model to the SpectraAI dashboard.")


if __name__ == "__main__":
    main()
