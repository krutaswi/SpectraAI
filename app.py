
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image

from image_features import extract_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "quality_model.pkl"

st.set_page_config(
    page_title="SpectraAI | Image Intelligence",
    page_icon="🛰️",
    layout="wide",
)


# ---------- Styling ----------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #07111f, #101b32);
        color: #e8f0ff;
    }

    h1, h2, h3 {
        color: #dbeafe;
    }

    .hero {
        padding: 24px;
        border: 1px solid #293d60;
        border-radius: 18px;
        background: linear-gradient(120deg, #132747, #17203b);
        margin-bottom: 20px;
    }

    .metric-card {
        background: #121f35;
        border: 1px solid #2b3d5d;
        border-radius: 14px;
        padding: 18px;
        min-height: 110px;
    }

    .metric-label {
        color: #a8b8d4;
        font-size: 14px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 700;
        color: #93c5fd;
    }

    .small-note {
        color: #a8b8d4;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header ----------

st.markdown(
    """
    <div class="hero">
        <h1>🛰️ SpectraAI</h1>
        <p>
            AI-powered image quality assessment for
            Earth observation and remote sensing.
        </p>
        <p class="small-note">
            IMAGE INTELLIGENCE · COMPUTER VISION · MACHINE LEARNING
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------- Model loading ----------

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()

try:
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()


# ---------- Upload ----------

st.subheader("01 / Upload an image")

uploaded_file = st.file_uploader(
    "Choose a satellite, aerial, or other image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is None:
    st.info(
        "Upload an image to analyze its visual characteristics "
        "and estimate its quality category."
    )

    st.markdown(
        """
        ### What SpectraAI measures

        - **Sharpness:** a measure of fine-detail variation
        - **Contrast:** variation in pixel intensity
        - **Brightness:** average grayscale intensity
        - **Edge density:** proportion of detected edge pixels
        - **Noise indicator:** high-frequency image variation
        """
    )
    st.stop()


# ---------- Image processing ----------

try:
    pil_image = Image.open(uploaded_file)

    # Display the uploaded image in RGB
    display_image = pil_image.convert("RGB")

    # Convert to OpenCV's BGR format for feature extraction
    image = cv2.cvtColor(
        np.array(display_image),
        cv2.COLOR_RGB2BGR,
    )

    image = cv2.resize(image, (256, 256))

except Exception as exc:
    st.error(f"Could not process this image: {exc}")
    st.stop()


features = extract_features(image)
feature_vector = np.array(
    [[features[name] for name in feature_names]],
    dtype=np.float64,
)

prediction = model.predict(feature_vector)[0]
probabilities = model.predict_proba(feature_vector)[0]

class_probabilities = dict(
    zip(model.classes_, probabilities)
)

confidence = float(np.max(probabilities)) * 100


# ---------- Results ----------

st.divider()
st.subheader("02 / Analysis results")

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.image(
        display_image,
        caption="Uploaded image",
        use_container_width=True,
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Predicted quality category</div>
            <div class="metric-value">{prediction}</div>
            <div class="small-note">
                Model confidence: {confidence:.1f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Confidence reflects the model's predicted class probabilities, "
        "not verified scientific accuracy."
    )

with right:
    st.markdown("### Image quality indicators")

    metric_labels = {
        "brightness": "Brightness",
        "contrast": "Contrast",
        "sharpness": "Sharpness",
        "edge_density": "Edge density",
        "noise_indicator": "Noise indicator",
    }

    for key, label in metric_labels.items():
        value = features[key]

        if key == "edge_density":
            display_value = f"{value * 100:.2f}%"
        else:
            display_value = f"{value:.2f}"

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display_value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(
            min(max(
                value / (
                    1.0 if key == "edge_density"
                    else 255.0 if key in ["brightness", "contrast"]
                    else max(value, 1.0)
                ),
                0.0,
            ), 1.0)
        )


# ---------- Probability breakdown ----------

st.divider()
st.subheader("03 / Model prediction breakdown")

for label, probability in class_probabilities.items():
    st.write(f"**{label}** — {probability:.1%}")
    st.progress(float(probability))


# ---------- Recommendations ----------

st.divider()
st.subheader("04 / Suggested next steps")

recommendations = []

if features["sharpness"] < 100:
    recommendations.append(
        "Inspect for blur. Consider cautious sharpening if appropriate."
    )

if features["contrast"] < 35:
    recommendations.append(
        "Consider contrast enhancement for better visual interpretation."
    )

if features["brightness"] < 60:
    recommendations.append(
        "The image is relatively dark. Review exposure and illumination."
    )
elif features["brightness"] > 200:
    recommendations.append(
        "The image is relatively bright. Check for overexposure or saturation."
    )

if features["noise_indicator"] > 50:
    recommendations.append(
        "Inspect high-frequency variation. Test denoising carefully "
        "to avoid removing genuine fine detail."
    )

if not recommendations:
    recommendations.append(
        "No obvious issues were flagged by these basic thresholds. "
        "This does not establish scientific suitability."
    )

for item in recommendations:
    st.write(f"- {item}")


# ---------- Scientific limitations ----------

with st.expander("About this model and its limitations"):
    st.write(
        "This prototype uses a Random Forest trained on synthetic "
        "degradations of a single source image. Its labels are generated "
        "from the degradation process, not expert assessments. "
        "Predictions may not generalize to unseen satellite imagery."
    )
    st.write(
        "The current version analyzes grayscale visual features. "
        "It does not perform calibrated atmospheric correction, "
        "reliable cloud detection, sensor-specific quality assessment, "
        "or scientific validation."
    )

st.caption("SpectraAI · Research prototype · Built with Python, OpenCV and scikit-learn")
