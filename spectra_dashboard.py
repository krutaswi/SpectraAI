
from pathlib import Path
from io import BytesIO
import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import rasterio
from rasterio.io import MemoryFile
from PIL import Image

from image_features import extract_features


# ============================================================
# SPECTRAAI — IMAGE QUALITY INTELLIGENCE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "data" / "metadata" / "quality_assessment.csv"
)


# Conservative limits for public hosting. These are checked before raster
# arrays are loaded into memory.
MAX_IMAGE_UPLOAD_MB = 20
MAX_IMAGE_PIXELS = 16_000_000
MAX_RASTER_UPLOAD_MB = 50
MAX_RASTER_TOTAL_MB = 100
MAX_RASTER_PIXELS = 2_000_000


def validate_upload_size(uploaded_file, max_mb, label="File"):
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"{label} is {size_mb:.1f} MB. The limit is {max_mb} MB."
        )


def validate_raster_dimensions(src, filename="Raster"):
    pixels = int(src.width) * int(src.height)
    if pixels > MAX_RASTER_PIXELS:
        raise ValueError(
            f"{filename} is {src.width:,} × {src.height:,} "
            f"({pixels:,} pixels). The current hosted-analysis limit is "
            f"{MAX_RASTER_PIXELS:,} pixels per raster."
        )
    if src.width < 1 or src.height < 1:
        raise ValueError(f"{filename} has invalid dimensions.")

st.set_page_config(
    page_title="SpectraAI | Image Quality Intelligence",
    page_icon="🛰️",
    layout="wide"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root { --ink:#e8f0f5; --muted:#9db0bd; --panel:#142536; --panel2:#192e40; --teal:#36d6c3; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background: radial-gradient(ellipse at 12% 0%, rgba(29,115,125,.20), transparent 34%), linear-gradient(145deg,#081521 0%,#0b1b29 52%,#0d2230 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(8,21,33,.78); }
    [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1d2b,#102638); border-right:1px solid rgba(130,180,190,.13); }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c5d3dc; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; color:#f2f7fa; }
    h1 { font-weight:700 !important; }
    h2,h3 { font-weight:600 !important; }
    p, label, .stCaption { color:#bdcbd4; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,rgba(25,46,64,.94),rgba(17,37,53,.94)); border:1px solid rgba(115,171,180,.18); padding:17px 19px; border-radius:16px; box-shadow:0 10px 28px rgba(0,0,0,.13); }
    [data-testid="stMetricLabel"] { color:#a9bdc8 !important; }
    [data-testid="stMetricValue"] { color:#f2fbfc !important; font-family:'Space Grotesk',sans-serif; }
    [data-testid="stTabs"] [role="tablist"] { gap:8px; border-bottom:1px solid rgba(126,174,185,.18); }
    [data-testid="stTabs"] button[role="tab"] { background:rgba(22,43,59,.72); border:1px solid rgba(126,174,185,.15); border-radius:11px 11px 0 0; padding:11px 18px; color:#b8c9d2; }
    [data-testid="stTabs"] button[aria-selected="true"] { background:rgba(39,130,132,.20); color:#66e3d4; border-color:rgba(54,214,195,.42); }
    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stPlotlyChart"]), div[data-testid="stImage"] { border-radius:14px; }
    [data-testid="stFileUploader"] section { background:rgba(19,42,58,.78); border:1px dashed rgba(71,194,185,.45); border-radius:15px; }
    .stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid rgba(54,214,195,.45); background:linear-gradient(135deg,#167d7b,#176b78); color:white; font-weight:600; padding:.55rem 1rem; transition:all .18s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:#69e8d8; background:linear-gradient(135deg,#19958e,#197d8c); transform:translateY(-1px); }
    [data-testid="stAlert"] { border-radius:12px; }
    [data-testid="stDataFrame"] { border:1px solid rgba(126,174,185,.16); border-radius:13px; overflow:hidden; }
    hr { border-color:rgba(126,174,185,.17) !important; }
    .spectra-hero { position:relative; overflow:hidden; padding:30px 32px; border:1px solid rgba(79,205,194,.25); border-radius:22px; background:radial-gradient(circle at 85% 25%,rgba(41,184,174,.22),transparent 27%),linear-gradient(115deg,rgba(18,42,59,.98),rgba(12,32,47,.96)); box-shadow:0 18px 45px rgba(0,0,0,.20); margin:4px 0 24px; }
    .spectra-kicker { color:#62dccc; font-size:11px; font-weight:700; letter-spacing:.19em; text-transform:uppercase; }
    .spectra-title { color:#f3fbfc; font-family:'Space Grotesk',sans-serif; font-size:clamp(34px,5vw,52px); font-weight:700; line-height:1.08; margin:8px 0; letter-spacing:-.045em; }
    .spectra-subtitle { color:#b6cbd5; font-size:15px; max-width:720px; line-height:1.65; }
    .spectra-chip { display:inline-block; margin-top:16px; margin-right:8px; padding:6px 10px; border-radius:999px; border:1px solid rgba(91,211,197,.25); background:rgba(39,142,139,.13); color:#9ce9df; font-size:11px; font-weight:600; }
    @media (max-width: 700px) { .spectra-hero { padding:23px 20px; } }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def quality_category(score):
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 50:
        return "Fair"
    return "Poor"


def safe_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    return float(np.mean(values)) if values.size else np.nan


def safe_std(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    return float(np.std(values)) if values.size else np.nan


# Consistent dark, presentation-ready styling for every Matplotlib chart.
plt.rcParams.update({
    "figure.facecolor": "#07142b",
    "axes.facecolor": "#0a1a35",
    "axes.edgecolor": "#35577e",
    "axes.labelcolor": "#c8d8f2",
    "axes.titlecolor": "#f3f7ff",
    "xtick.color": "#a9bfdc",
    "ytick.color": "#a9bfdc",
    "text.color": "#eaf2ff",
    "grid.color": "#294363",
    "grid.alpha": 0.48,
    "font.family": "DejaVu Sans",
    "axes.prop_cycle": plt.cycler(color=["#43d9ef", "#9b7bff", "#55d6b0", "#ffb86b", "#f27ca8"]),
})


def render_chart(fig, container=None):
    """Apply SpectraAI's dark chart theme before rendering a Matplotlib figure."""
    for ax in fig.axes:
        ax.set_facecolor("#0a1a35")
        ax.title.set_color("#f3f7ff")
        ax.xaxis.label.set_color("#c8d8f2")
        ax.yaxis.label.set_color("#c8d8f2")
        ax.tick_params(colors="#a9bfdc", labelsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.65, alpha=0.42, color="#355477")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("#35577e")
            spine.set_alpha(0.65)
    fig.patch.set_facecolor("#07142b")
    target = container if container is not None else st
    target.pyplot(fig, use_container_width=True)


NAV_LABELS = {
    "Mission Overview": "⌂  Mission Overview",
    "Observations": "◇  Observations",
    "Spectral Lab": "▥  Spectral Lab",
    "Quality Control": "⬡  Quality Control",
    "Anomaly Lab": "◎  Anomaly Lab",
    "Image Analysis": "▧  Image Analysis",
}


def navigate_to(page):
    """Switch the sidebar radio to a workspace from a Mission Overview tile."""
    st.session_state["mission_navigation"] = NAV_LABELS[page]


# ============================================================
# DATASET LOADING
# ============================================================

@st.cache_data
def load_data():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Quality assessment file not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        dtype={
            "area_id": str,
            "date_doy": str,
            "tile": str
        }
    )

    df["area_id"] = df["area_id"].str.zfill(4)

    for column in [
        "quality_score",
        "high_aerosol_pct",
        "clear_reflectance_coverage_pct"
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# ORDINARY IMAGE ANALYSIS
# ============================================================

def analyse_ordinary_image(image_array):

    features = extract_features(
        image_array,
        color_order="RGB"
    )

    brightness = features["brightness"]
    contrast = features["contrast"]
    sharpness = features["sharpness"]

    # These are heuristic component scores, not trained predictions.
    brightness_score = max(
        0,
        100 - abs(brightness - 127.5) / 127.5 * 100
    )

    contrast_score = min(
        100,
        contrast / 60 * 100
    )

    sharpness_score = min(
        100,
        100 * math.log1p(sharpness) / math.log1p(500)
    )

    overall_score = (
        0.30 * brightness_score
        + 0.25 * contrast_score
        + 0.45 * sharpness_score
    )

    overall_score = float(np.clip(overall_score, 0, 100))

    return {
        **features,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "sharpness_score": sharpness_score,
        "quality_score": overall_score,
        "quality_category": quality_category(overall_score)
    }


# ============================================================
# EXPERIMENTAL VISUAL ANOMALY SCAN
# ============================================================

def detect_visual_outliers(image_array):
    """Highlight pixels that differ strongly from their local surroundings.

    This is an unsupervised image-processing heuristic. It does not classify
    wildfire, damage, or any other real-world event.
    """
    rgb = np.asarray(image_array, dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Estimate the local background and measure each pixel's colour distance.
    background = cv2.GaussianBlur(lab, (0, 0), sigmaX=24, sigmaY=24)
    difference = np.linalg.norm(lab - background, axis=2)

    values = difference[np.isfinite(difference)]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_threshold = median + 3.5 * 1.4826 * mad
    threshold = max(10.0, robust_threshold)

    raw_mask = (difference > threshold).astype(np.uint8)

    # Remove isolated speckles and join nearby pixels into coherent regions.
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    minimum_area = max(12, int(rgb.shape[0] * rgb.shape[1] * 0.00002))
    final_mask = np.zeros(cleaned.shape, dtype=np.uint8)
    regions = []

    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_area:
            final_mask[labels == label] = 255
            regions.append({
                "x": int(x), "y": int(y),
                "width": int(width), "height": int(height),
                "area_pixels": int(area)
            })

    anomaly_pct = 100.0 * np.count_nonzero(final_mask) / final_mask.size

    overlay = rgb.copy()
    tint = np.zeros_like(overlay)
    tint[:, :] = (255, 45, 20)  # orange-red highlight in RGB
    selected = final_mask > 0
    overlay[selected] = (
        0.45 * overlay[selected] + 0.55 * tint[selected]
    ).astype(np.uint8)

    return {
        "mask": final_mask,
        "overlay": overlay,
        "anomaly_pct": float(anomaly_pct),
        "region_count": len(regions),
        "regions": regions,
        "threshold": float(threshold)
    }


# ============================================================
# SATELLITE RASTER READING
# ============================================================

def read_uploaded_raster(uploaded_file, band_number=1):

    validate_upload_size(
        uploaded_file, MAX_RASTER_UPLOAD_MB, "GeoTIFF"
    )

    with MemoryFile(uploaded_file.getvalue()) as memory_file:

        with memory_file.open() as src:

            validate_raster_dimensions(src, uploaded_file.name)

            if band_number < 1 or band_number > src.count:
                raise ValueError(
                    f"Band {band_number} does not exist in "
                    f"{uploaded_file.name}."
                )

            raw = src.read(band_number)

            valid = src.read_masks(band_number) > 0

            nodata = src.nodata

            if nodata is not None:
                valid &= raw != nodata

            scale = (
                src.scales[band_number - 1]
                if src.scales else 1.0
            )

            offset = (
                src.offsets[band_number - 1]
                if src.offsets else 0.0
            )

            scale = 1.0 if scale is None else float(scale)
            offset = 0.0 if offset is None else float(offset)

            metadata = {
                "width": src.width,
                "height": src.height,
                "crs": src.crs,
                "transform": src.transform,
                "count": src.count
            }

            return raw, valid, scale, offset, metadata


def detect_hls_layer(filename):

    name = filename.upper()

    for layer in ["B02", "B03", "B04", "B8A"]:
        if layer in name:
            return layer

    if "FMASK" in name or "FMASK" in name.replace("_", ""):
        return "Fmask"

    return None


def validate_raster_grids(metadata_list):

    reference = metadata_list[0]

    for metadata in metadata_list[1:]:

        if metadata["width"] != reference["width"]:
            raise ValueError("Raster width mismatch.")

        if metadata["height"] != reference["height"]:
            raise ValueError("Raster height mismatch.")

        if metadata["crs"] != reference["crs"]:
            raise ValueError("Raster coordinate systems do not match.")

        if metadata["transform"] != reference["transform"]:
            raise ValueError("Raster spatial transforms do not match.")


# ============================================================
# SATELLITE QUALITY ANALYSIS
# ============================================================

def analyse_satellite_arrays(bands, band_masks, fmask, fmask_valid):

    CLOUD_BIT = 2
    ADJACENT_BIT = 4
    SHADOW_BIT = 8
    SNOW_BIT = 16
    WATER_BIT = 32

    AEROSOL_HIGH = 192

    total_pixels = fmask.size

    common_valid = (
        band_masks["B02"]
        & band_masks["B03"]
        & band_masks["B04"]
        & band_masks["B8A"]
        & fmask_valid
    )

    if not np.any(common_valid):
        raise ValueError(
            "No pixels are valid across all uploaded layers."
        )

    valid_fmask_values = fmask[fmask_valid].astype(np.uint8)

    def flag_percentage(bit):
        if valid_fmask_values.size == 0:
            return 0.0

        return float(
            100 * np.mean(
                (valid_fmask_values & bit) != 0
            )
        )

    cloud_pct = flag_percentage(CLOUD_BIT)
    adjacency_pct = flag_percentage(ADJACENT_BIT)
    shadow_pct = flag_percentage(SHADOW_BIT)
    snow_pct = flag_percentage(SNOW_BIT)
    water_pct = flag_percentage(WATER_BIT)

    aerosol_classes = valid_fmask_values & 192

    high_aerosol_pct = float(
        100 * np.mean(aerosol_classes == AEROSOL_HIGH)
    )

    excluded_flags = (
        CLOUD_BIT
        | ADJACENT_BIT
        | SHADOW_BIT
        | SNOW_BIT
    )

    clear_fmask = (
        fmask_valid
        & ((fmask.astype(np.uint8) & excluded_flags) == 0)
    )

    clear_mask = common_valid & clear_fmask

    clear_pixels = int(np.count_nonzero(clear_mask))

    quality_score = (
        100 * clear_pixels / total_pixels
        if total_pixels else 0
    )

    if clear_pixels == 0:
        raise ValueError(
            "No clear reflectance pixels were found. "
            "The uploaded scene may be fully masked or invalid."
        )

    bands_stats = {}

    for band in ["B02", "B03", "B04", "B8A"]:
        bands_stats[band] = safe_mean(bands[band][clear_mask])

    blue = bands["B02"]
    green = bands["B03"]
    red = bands["B04"]
    nir = bands["B8A"]

    ndvi_denominator = nir + red
    ndwi_denominator = green + nir

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndwi = np.full(red.shape, np.nan, dtype=np.float32)

    np.divide(
        nir - red,
        ndvi_denominator,
        out=ndvi,
        where=clear_mask & (np.abs(ndvi_denominator) > 1e-6)
    )

    np.divide(
        green - nir,
        ndwi_denominator,
        out=ndwi,
        where=clear_mask & (np.abs(ndwi_denominator) > 1e-6)
    )

    evi_denominator = nir + 6 * red - 7.5 * blue + 1

    evi = np.full(red.shape, np.nan, dtype=np.float32)

    np.divide(
        2.5 * (nir - red),
        evi_denominator,
        out=evi,
        where=clear_mask & (np.abs(evi_denominator) > 1e-6)
    )

    results = {
        "quality_score": float(quality_score),
        "quality_category": quality_category(quality_score),
        "clear_reflectance_coverage_pct": float(quality_score),
        "clear_pixels_pct": float(
            100 * clear_pixels / np.count_nonzero(fmask_valid)
        ),
        "reflectance_valid_coverage_pct": float(
            100 * np.count_nonzero(common_valid) / total_pixels
        ),
        "cloud_pct": cloud_pct,
        "adjacent_cloud_shadow_pct": adjacency_pct,
        "cloud_shadow_pct": shadow_pct,
        "snow_ice_pct": snow_pct,
        "water_pct": water_pct,
        "high_aerosol_pct": high_aerosol_pct,
        "NDVI_mean": safe_mean(ndvi[clear_mask]),
        "NDWI_mean": safe_mean(ndwi[clear_mask]),
        "EVI_mean": safe_mean(evi[clear_mask]),
        "band_means": bands_stats,
        "total_pixels": int(total_pixels),
        "clear_pixels": clear_pixels
    }

    return results


def detect_spectral_anomalies(bands, band_masks, fmask, fmask_valid):
    """Flag multispectral pixels with unusual four-band signatures.

    Uses robust per-band median/MAD scaling over valid, clear pixels.
    This is an unsupervised statistical screening method, not a trained
    event classifier; candidates may represent real surface variation,
    mixed pixels, residual atmospheric effects, or data artefacts.
    """
    excluded_flags = 2 | 4 | 8 | 16
    common_valid = (
        band_masks["B02"] & band_masks["B03"]
        & band_masks["B04"] & band_masks["B8A"] & fmask_valid
    )
    clear_mask = (
        common_valid
        & ((fmask.astype(np.uint8) & excluded_flags) == 0)
    )

    if np.count_nonzero(clear_mask) < 20:
        raise ValueError("Too few clear pixels for spectral anomaly screening.")

    band_names = ["B02", "B03", "B04", "B8A"]
    z_layers = []
    for name in band_names:
        values = np.asarray(bands[name], dtype=np.float32)
        sample = values[clear_mask]
        median = float(np.median(sample))
        mad = float(np.median(np.abs(sample - median)))
        robust_scale = 1.4826 * mad
        if robust_scale < 1e-6:
            robust_scale = float(np.std(sample))
        robust_scale = max(robust_scale, 1e-4)
        z = np.zeros(values.shape, dtype=np.float32)
        z[clear_mask] = (values[clear_mask] - median) / robust_scale
        z_layers.append(z)

    spectral_distance = np.sqrt(np.mean(np.stack(z_layers, axis=0) ** 2, axis=0))
    raw_mask = ((spectral_distance > 3.5) & clear_mask).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    minimum_area = max(4, int(cleaned.size * 0.00001))
    final_mask = np.zeros(cleaned.shape, dtype=np.uint8)
    regions = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area >= minimum_area:
            final_mask[labels == label] = 255
            regions.append({
                "x": int(x), "y": int(y), "width": int(width),
                "height": int(height), "area_pixels": int(area),
                "mean_spectral_distance": float(np.mean(spectral_distance[labels == label]))
            })

    # Build a percentile-stretched natural-colour preview from B04/B03/B02.
    rgb = np.zeros((*final_mask.shape, 3), dtype=np.uint8)
    for channel, name in enumerate(["B04", "B03", "B02"]):
        values = np.asarray(bands[name], dtype=np.float32)
        sample = values[clear_mask]
        low, high = np.percentile(sample, [2, 98])
        if high <= low:
            high = low + 1e-4
        scaled = np.clip((values - low) / (high - low), 0, 1)
        rgb[..., channel] = (scaled * 255).astype(np.uint8)

    overlay = rgb.copy()
    selected = final_mask > 0
    overlay[selected] = (
        0.45 * overlay[selected] + 0.55 * np.array([255, 40, 20])
    ).astype(np.uint8)

    anomaly_pct = 100.0 * np.count_nonzero(final_mask) / max(1, np.count_nonzero(clear_mask))
    return {
        "mask": final_mask,
        "overlay": overlay,
        "anomaly_pct": float(anomaly_pct),
        "region_count": len(regions),
        "regions": regions,
        "clear_pixels": int(np.count_nonzero(clear_mask))
    }


def show_spectral_anomaly_results(bands, band_masks, fmask, fmask_valid, key_prefix):
    st.divider()
    st.subheader("Multispectral Anomaly Screening")
    st.caption(
        "Screens clear pixels for unusual combinations of Blue, Green, Red "
        "and NIR reflectance using robust statistical outlier scores. "
        "Highlighted pixels are candidates for inspection, not confirmed "
        "wildfire, damage, or land-change detections."
    )
    anomaly = detect_spectral_anomalies(bands, band_masks, fmask, fmask_valid)
    col1, col2 = st.columns(2)
    col1.metric("Candidate Anomaly Area", f"{anomaly['anomaly_pct']:.3f}% of clear pixels")
    col2.metric("Candidate Regions", anomaly["region_count"])
    st.image(anomaly["overlay"], caption="Natural-colour preview with spectral outlier candidates highlighted in orange-red", use_container_width=True)
    if anomaly["regions"]:
        st.dataframe(pd.DataFrame(anomaly["regions"]), use_container_width=True, hide_index=True)
    else:
        st.info("No regions passed the current statistical screening threshold. This does not prove the scene is anomaly-free.")
    mask_buffer = BytesIO()
    Image.fromarray(anomaly["mask"]).save(mask_buffer, format="PNG")
    st.download_button(
        "Download Spectral Anomaly Mask", mask_buffer.getvalue(),
        file_name="spectraai_spectral_anomaly_mask.png", mime="image/png",
        key=f"{key_prefix}_spectral_mask"
    )
    anomaly_report = pd.DataFrame([{
        "candidate_anomaly_area_pct_of_clear_pixels": anomaly["anomaly_pct"],
        "candidate_region_count": anomaly["region_count"],
        "clear_pixels_screened": anomaly["clear_pixels"]
    }])
    st.download_button(
        "Download Spectral Anomaly Report", anomaly_report.to_csv(index=False).encode("utf-8"),
        file_name="spectraai_spectral_anomaly_report.csv", mime="text/csv",
        key=f"{key_prefix}_spectral_report"
    )


# ============================================================
# MISSION CONTROL NAVIGATION
# ============================================================

COSMOS_BG_DATA = "data:image/webp;base64,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"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--cyan:#53e4f4;--blue:#72b8f4;--violet:#a987ff;--white:#f4f7ff;--muted:#b8c8e3;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--white);}
.stApp{background:#030817 url('__COSMOS_BG__') center center/cover fixed no-repeat;}
[data-testid="stAppViewContainer"]{background:linear-gradient(90deg,rgba(2,7,20,.22),rgba(2,7,20,.05) 55%,rgba(2,7,20,.18));}
[data-testid="stHeader"]{height:0;background:transparent;}
[data-testid="stToolbar"]{right:1rem;}
[data-testid="stSidebar"]{width:242px!important;min-width:242px!important;background:linear-gradient(180deg,rgba(3,9,24,.97),rgba(5,15,36,.94));border-right:1px solid rgba(93,190,245,.30);box-shadow:12px 0 40px rgba(0,0,0,.2);}
[data-testid="stSidebar"]>div{padding-top:1rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{color:#b9c9e3;}
[data-testid="stSidebar"] [data-testid="stRadio"]>label{display:none;}
[data-testid="stSidebar"] [data-testid="stRadio"]>div{gap:7px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label{padding:9px 12px;border:1px solid transparent;border-radius:12px;color:#c4d2ec;font-size:13px;min-height:42px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:rgba(64,172,222,.13);border-color:rgba(90,206,239,.18);}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(100deg,rgba(35,174,220,.34),rgba(35,93,156,.28));border-color:rgba(76,221,250,.65);box-shadow:0 0 22px rgba(35,180,232,.18);}
[data-testid="stSidebar"] [data-testid="stRadio"] label p{color:inherit!important;}
.main .block-container{max-width:none;padding:1.05rem 1.6rem 1.2rem 1.8rem;}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#f5f8ff!important;}
p,label,.stCaption{color:#c5d2e8;}
[data-testid="stMetric"]{background:rgba(5,17,41,.55)!important;border:1px solid rgba(92,177,239,.30)!important;border-radius:14px!important;backdrop-filter:blur(16px);box-shadow:0 12px 30px rgba(0,0,0,.22)!important;padding:12px 15px;}
[data-testid="stMetricLabel"]{color:#a9c6e8!important;font-size:10px!important;}
[data-testid="stMetricValue"]{color:#f7f9ff!important;font-family:'Space Grotesk',sans-serif!important;}
.stButton>button,.stDownloadButton>button{border-radius:999px;border:1px solid rgba(76,220,245,.62);background:linear-gradient(100deg,rgba(12,103,143,.88),rgba(48,73,148,.9));color:#f7fbff;font-weight:600;}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#a3f3ff;box-shadow:0 0 22px rgba(65,205,246,.25);}
[data-testid="stFileUploader"] section{background:rgba(8,20,45,.84);border:1px dashed rgba(105,190,244,.48);border-radius:14px;}
[data-testid="stDataFrame"]{border:1px solid rgba(110,174,235,.25);border-radius:12px;overflow:hidden;}
[data-testid="stImage"] img{border-radius:13px;}
hr{border-color:rgba(116,177,238,.20)!important;}
/* Sidebar brand */
.mc-brand-lockup{text-align:center;padding:12px 5px 26px;border-radius:0 0 25px 25px;background:radial-gradient(ellipse at 50% 8%,rgba(34,138,221,.22),transparent 65%);margin-bottom:20px;}
.mc-planet{font-size:45px;line-height:1;color:#5ee8ff;text-shadow:0 0 18px #2eaeff,0 0 34px rgba(77,129,255,.7);transform:rotate(-20deg);display:inline-block;}
.mc-brand-name{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:600;letter-spacing:-.04em;color:#f5f8ff;margin-top:3px;}
.mc-brand-sub{font-size:9px;letter-spacing:.17em;line-height:1.5;color:#b7c9e7;margin-top:3px;}
.mc-nav-label{font-size:9px;letter-spacing:.19em;color:#7399c6;text-transform:uppercase;margin:0 0 8px 9px;}
.mc-sidebar-footer{position:relative;margin-top:22px;padding:15px 10px 8px;border-top:1px solid rgba(111,158,218,.22);font-size:9px;letter-spacing:.12em;color:#9cb8df;}
/* Reference composition */
.mc-topline{display:flex;justify-content:space-between;align-items:center;margin:0 2px 16px;font-size:10px;color:#e0eaff;}
.mc-dataset-pill{display:inline-flex;align-items:center;gap:9px;padding:8px 14px;border:1px solid rgba(90,183,241,.38);border-radius:999px;background:rgba(3,14,35,.52);backdrop-filter:blur(14px);}
.mc-pulse{width:9px;height:9px;border-radius:50%;background:#37e7c0;box-shadow:0 0 12px #37e7c0;display:inline-block;}
.mc-top-right{font-size:10px;letter-spacing:.08em;color:#cbd9f1;}
.mc-hero{display:grid;grid-template-columns:minmax(0,1fr) 238px;gap:24px;align-items:start;min-height:320px;padding:48px 0 18px 5px;}
.mc-welcome-small{font-family:'Space Grotesk',sans-serif;font-size:27px;line-height:1.1;color:#edf3ff;}
.mc-welcome-big{font-family:'Space Grotesk',sans-serif;font-size:68px;line-height:.98;font-weight:700;letter-spacing:-.065em;color:#f7f9ff;margin:0 0 14px;text-shadow:0 3px 24px rgba(0,0,0,.35);}
.mc-welcome-big span{color:#53dff4;}
.mc-tagline{font-size:12px;letter-spacing:.22em;color:#aeeaff;font-weight:600;}
.mc-hero-rule{height:3px;width:130px;background:linear-gradient(90deg,#55e8f0,transparent);margin:22px 0;box-shadow:0 0 14px rgba(62,224,255,.5);}
.mc-hero-copy{font-size:14px;line-height:1.55;color:#e0e9fa;max-width:440px;text-shadow:0 2px 10px #030817;}
.mc-status-card{margin-top:2px;padding:18px 19px 13px;border:1px solid rgba(91,190,248,.55);border-radius:16px;background:linear-gradient(145deg,rgba(4,16,39,.86),rgba(4,13,33,.78));backdrop-filter:blur(16px);box-shadow:0 15px 40px rgba(0,0,0,.28),inset 0 1px rgba(255,255,255,.06);}
.mc-status-title{font-size:13px;font-weight:700;color:#f4f7ff;margin-bottom:14px;}
.mc-status-row{display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid rgba(140,179,232,.15);}
.mc-status-row:last-child{border-bottom:0;}
.mc-status-icon{font-size:22px;color:#5adff1;width:28px;text-align:center;}
.mc-status-label{font-size:11px;color:#dce8fb;line-height:1.3;}
.mc-status-label b{display:block;color:#f2f6ff;font-size:11px;font-weight:600;margin-bottom:2px;}
.mc-panel{border:1px solid rgba(83,182,243,.55);border-radius:17px;background:linear-gradient(120deg,rgba(4,15,36,.80),rgba(3,13,32,.77));backdrop-filter:blur(15px);box-shadow:0 15px 38px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.05);}
.mc-section-title{font-family:'Space Grotesk',sans-serif;font-size:18px;font-weight:600;color:#f3f7ff;margin:0 0 12px;}
.mc-section-title span{color:#53e5f5;margin-right:8px;font-size:21px;}
.mc-mission{display:grid;grid-template-columns:263px minmax(230px,1fr) 1.15fr;gap:22px;align-items:center;padding:15px 19px 17px;min-height:194px;}
.mc-mission-image{width:100%;height:119px;object-fit:cover;border-radius:12px;border:1px solid rgba(89,186,244,.42);}
.mc-mission-copy{font-size:12px;line-height:1.6;color:#d1def2;}
.mc-explore{display:inline-block;margin-top:13px;padding:7px 16px;border:1px solid #42d9f2;border-radius:999px;color:#dffaff;font-size:11px;background:rgba(10,68,102,.35);}
.mc-metrics{display:grid;grid-template-columns:repeat(3,1fr);align-items:center;border-left:1px solid rgba(138,185,237,.45);min-height:126px;}
.mc-metric{text-align:center;border-right:1px solid rgba(138,185,237,.27);padding:5px 8px;}
.mc-metric:last-child{border-right:0;}
.mc-metric-icon{font-size:23px;color:#4fe3f3;line-height:1.2;}
.mc-metric-value{font-family:'Space Grotesk',sans-serif;font-size:25px;font-weight:700;color:#f7f9ff;margin-top:3px;}
.mc-metric-label{font-size:11px;color:#c6d4eb;}
.mc-lower{display:grid;grid-template-columns:minmax(0,3fr) minmax(250px,.92fr);gap:24px;margin-top:25px;}
.mc-quick-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:13px 17px;}
.mc-quick-card{height:136px;position:relative;overflow:hidden;border:1px solid rgba(94,174,240,.50);border-radius:15px;background-color:rgba(4,14,34,.78);background-size:cover;background-position:center;box-shadow:0 9px 24px rgba(0,0,0,.22);}
.mc-quick-card:before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(3,11,29,.94) 0%,rgba(3,12,30,.79) 53%,rgba(3,12,30,.12) 100%);}
.mc-quick-text{position:absolute;left:16px;top:18px;z-index:1;max-width:72%;}
.mc-quick-text b{display:block;font-family:'Space Grotesk',sans-serif;font-size:14px;color:#f5f8ff;margin-bottom:5px;}
.mc-quick-text small{font-size:11px;line-height:1.35;color:#e0eafa;}
.mc-quick-arrow{position:absolute;right:12px;bottom:12px;width:30px;height:30px;border:1px solid #47d9f4;border-radius:50%;display:flex;align-items:center;justify-content:center;color:#53e6f6;font-size:17px;background:rgba(3,18,43,.76);z-index:2;}
.st-key-quick_access_cards [data-testid="stHorizontalBlock"]{align-items:stretch!important;gap:14px!important;}
.st-key-quick_access_cards [data-testid="column"]{min-width:0!important;}
.st-key-quick_access_cards [data-testid="stButton"]{width:100%;}
.st-key-quick_access_cards [data-testid="stButton"]>button{box-sizing:border-box;width:100%;height:120px;min-height:120px;white-space:pre-line;text-align:left;display:flex;align-items:flex-start;justify-content:flex-start;padding:16px 15px;border:1px solid rgba(94,174,240,.55);border-radius:15px;background:linear-gradient(120deg,rgba(4,17,42,.94),rgba(13,35,72,.88));box-shadow:0 9px 22px rgba(0,0,0,.22),inset 0 1px rgba(255,255,255,.05);font-family:'Space Grotesk',sans-serif;font-size:14px;line-height:1.42;color:#f4f8ff;transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease;overflow:hidden;}
.st-key-quick_access_cards [data-testid="stButton"]>button:hover{transform:translateY(-2px);border-color:#58e4fa;background:linear-gradient(120deg,rgba(7,39,70,.98),rgba(35,40,91,.94));box-shadow:0 12px 26px rgba(34,174,235,.17);}
.st-key-quick_access_cards [data-testid="stButton"]>button:focus{box-shadow:0 0 0 2px rgba(80,220,245,.35);}
.mc-quote{height:254px;min-height:254px;box-sizing:border-box;border:1px solid rgba(99,173,238,.52);border-radius:16px;background-size:cover;background-position:center;position:relative;overflow:hidden;padding:22px 20px;}
.mc-quote:before{content:'';position:absolute;inset:0;background:linear-gradient(90deg,rgba(5,13,35,.78),rgba(5,13,35,.24)),linear-gradient(0deg,rgba(3,8,22,.28),transparent);}
.mc-quote-content{position:relative;z-index:1;}
.mc-quote-star{font-size:22px;color:#5de6f6;}
.mc-quote-text{font-family:'Space Grotesk',sans-serif;font-style:italic;font-size:13px;line-height:1.5;color:#f1f5ff;margin:11px 0 12px;max-width:260px;}
.mc-quote-author{font-size:10px;color:#b8c8e7;}
.mc-footnote{font-size:9px;color:#8ea9d1;margin-top:12px;}
@media(max-width:1100px){.mc-hero{grid-template-columns:1fr 220px;min-height:285px;padding-top:28px}.mc-welcome-big{font-size:54px}.mc-mission{grid-template-columns:210px 1fr}.mc-metrics{grid-column:1/-1;border-left:0;border-top:1px solid rgba(138,185,237,.3);padding-top:10px;min-height:80px}.mc-lower{grid-template-columns:1fr}.mc-quote{min-height:150px}}
@media(max-width:700px){.main .block-container{padding:1rem .8rem}.mc-hero{grid-template-columns:1fr;padding-top:15px}.mc-status-card{max-width:none}.mc-welcome-big{font-size:48px}.mc-mission{grid-template-columns:1fr}.mc-quick-grid{grid-template-columns:repeat(2,1fr)}.mc-top-right{display:none}}
</style>
""".replace('__COSMOS_BG__', COSMOS_BG_DATA), unsafe_allow_html=True)


# Reference image assets (cropped from the supplied visual reference)
MISSION_IMG = 'data:image/jpeg;base64,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'
OBS_IMG = 'data:image/jpeg;base64,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'
SPECTRAL_IMG = 'data:image/jpeg;base64,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'
QUALITY_IMG = 'data:image/jpeg;base64,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'
ANOMALY_IMG = 'data:image/jpeg;base64,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'
SPIRAL_IMG = 'data:image/jpeg;base64,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'
QUOTE_IMG = 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBAUEBAYFBQUGBgYHCQ4JCQgICRINDQoOFRIWFhUSFBQXGiEcFxgfGRQUHScdHyIjJSUlFhwpLCgkKyEkJST/2wBDAQYGBgkICREJCREkGBQYJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCQkJCT/wAARCAD0AVsDASIAAhEBAxEB/8QAHAAAAgMBAQEBAAAAAAAAAAAAAQIAAwQFBgcI/8QASBAAAQMDAgQCBgUICQMEAwAAAQACAwQFERIhBjFBURNhBxQicZHRFTKBk7MjJDM0UlN0oRZCQ2Jjc7HB4SVygwhUZPGCkvD/xAAZAQADAQEBAAAAAAAAAAAAAAAAAQIDBAX/xAAqEQACAgICAQMDBQADAAAAAAAAAQIRAyESMQQTQVEi8PEUMmFxkSOxwf/aAAwDAQACEQMRAD8A/MaiOEDsvTOcCYMV1DRy19QIYtI2LnPccNY0c3E9gugaukohooqdkjhzqKhoc53mGnZo+J804x92TKVaOTpB6j4qaB3HxW83etL/ANK3ftG35JX3esAOJm/dt+SPpC2YtI/aHxQLR3HxWgXitJyZh9235KfTNdnaYf8A6N+Sm4jpmfSOhHxU0juFebzXA/ph9235KfTVf+/H3bfklyiOmZ8DuFMAdR8Vo+ma79+Pu2/JD6Zrv3zfu2/JHKIUyjbuPipgZ5j4rT9M12P0zfu2/JD6Zrufit+7b8kcohTM+B3HxRw3uPir/pqv/fN+6Z8kRe6/H6Zv3TPknyiFMz6R3HxULR3HxWn6cr/37fumfJT6arz/AGzfumfJO4ipmXSPL4o6R3HxWj6Zrv3zfu2fJH6arv34+7Z8kXEKZn0t7j4ohoPUfFaPpmu/fN+7Z8lPpmv/AH4+7b8k+UQpmctbjmPihgZ5j4rV9MVuN5hn/Lb8lPpet/fD7tvyTuIqZlw3PMfFE4B5j4rV9MV375v3bPkj9NVuP0w+7b8k7iGzLkY5t+KOQOo+K0/TVd+/H3bPkj9NVh2MzfumfJNOIUzO0gdR8VYS3TzGffyVn0zWfvW/dM+Sdt7rf3rD/wCJnyVxlElpmV4AOxHxSEgDGR8Vvku9ZnaYY6fk2fJVfTleD+lZ90z5JS4oFZjwM5yPihgdx8VtN7rv3sZ/8LPkob1XbflI9/8ABZ8lP0j+oxaR3COkZ5hajeq7O8jPumfJA3qvB/TM+6Z8lNxQ/qMukdwppGOY+K1fTdf++b90z5Ii+V4z+WZ90z5IuP3+QqRl0juPiiGjuPitIv1wH9uz7lnyRF/uHWoby6RM+SalD7/Iql9/gz6BjmPioGDuPirxfrhz8dp/8TPknPEFxbjEzNxn9Ez5KlKH3+RVL7/Bl0DHNvxU0juPitf9ILhj9Ybq/wApnyQ/pDc//cD7tnyRyh9/kKl9/gwlLlMQgBusmaHUY71Syta3Z9bIS4/4bDgD3F2T9gWFxyt1S0ut9twM4hf+K5c1+xVT0kZrbGBzzQ8NxaXBp08s42+K000TIKcV0zY5Wh+lsTj9Y/3h2WOSqe5pYCWxl2rQ07ZSqlcilvorG223NLnBzsgVFhZoRRTCiQydEcIIpgBRFRAEUUwimBMfzUUUTEBHqgiOaADjZFTmplMQQpsopyTAPRTYjGN+6A80QcJgDKIGeiCPLbKBEx0zlMCht80cdeipCLBgt0kYPMYG596pPNMNztj7SlTbsEiEDbdBE+5Qbb9VIxTugmQO+5U0MA3PPCnvyphQbpADCiY88dkCihgxvsmILCWvaQ7sdiEo5q3x3YeMM9tukktyffnofNVGvcTK90ESSfsVrqWdhw6JzTgHBHQjIQot9BZClB3TlLhNiOjXSFlvtmNswP8AxXrnncZPT+a23EH1K1D/AOO/8V6573JzZMUR7sjHnlVFONyAc/YppBO3JYvZp0IASm0HstMMDng4aTgZOByC9YzhCnf6NJeJwJvWmXdtF9b2PCMec476uq0hibM55VHs8UW4S8lvnopooo5ZInsjlyY3ubgPxscd1je3BUyhRalYo57hBQoqCgIosY6R4YxrnE9AMkqHGSQE69xARQUwgA9VEcDCtpZzSzsmayN7mHIEjdTftHVUgKlMbqKIEFRDBwimASoEzpNUbGaWjRncDc5790XuYdOhmnAAO+cnqU6EJ0R6qHpuigAKIgKBMAKZ25o4QA3QAc7YURAQwgRDyU1Et09Aco43QI3TAGOagGtwAwM7bo+5B2MDHPqgYvdTAx1zlFD35UjIgR13R69VNyAM7DoigF96O2BuoNweSnJIYRz25K9kL5Wh+tu/c7rOjlaQkovaslr4LSgEeqHVKgNdzdijtY7U7/xXrnudq581uuf6pbP8h34r1z1GR7CK0HONxsU7BjdVp2OAKlDZ9U4E9DXE16nhrG1tFbqdzM+M2dszwCOXhtz35Fdy5Xqior/H6NPo6rksELfUZo2QfnEtSd/WWtHUO3A/Zyvk/D90FpuEdSKitpWtBJdRP0SE42GeWM4znovtzPSKHej1nH7qKi/pVCDaWVhZ7Won6x7+yM/8L0MLjX06PL8iM1Jctp6XtTPI8U+hDi62t1mqt9bTQRkRl9W2JzWDfAY8jHuHVfK3gbg9F2eJLgbnW+syXKouT5mNkkkqQdbXn6zTnbY9tsLiSFc+Zq9HZgUkvqdiEJUSoudnSWU1RLSzsngkdHKw5a9vMIx0004c6KN8mCM6RkgnlsqtgraeeWmkbNDI+KRp9l7Dgg+RVRfs+hP+CtzCxxa4EEHBB6KNBzsMonJJJ3UGRyOEgAid0COxR5IALiScnmlVjAZHaQBlx5lLzVABFRT3IERHHJHCgTAmEds8jhEDZQp0Imnb3qEI7Dmp36pgKphFEDfA3RQA6KYHNNgt35IEdk6EAbKEYV9JSTVk8dPBGZJZHaWsHNx7KuaN0crmSAtewlrgehCri6sL3RUSg5EjdBQMg58s4Q5oqY6qRgwoOaZwycgYQdjUcAgdj0Q0ACMOIOOfRAjZFQbHOM+9IYqKIOAdhvtlTHmnQFpCCbkgqJNNyAFHay4E5p3cj/ivXOI32XSuY/MrV/Dv/FeudjKma2OPQqmVMI4WZQ7HrpC+Vv0L9DeL+ZCp9b8PH9pp05z7hyXKGyYlXGTRLin2WOflVk5KGcqJN2NIBUwjhTHdTQCpgoQj0800gIoiphOgAojhTCAI0kEEEgjcEKdUQD7lMbZTEQ8lEd8BQpgMDtg8lMYCHPCboqQgDsiUdOFMbp0AFMBHG6KBCo5wjpzsECMJgEYzvnHkoCABkEn+SgChTQh/GeZA9p0EctO2FW9znuJcSSTkk8yj05boEJtt6ATCmDsnIS4U0OwYQA88BEqY25qRhD2iMt0ZcSCHZ5DthKWu0h2DpJxnG2VMKZONOTjnjOyBgx5o6Tqw058wmzjODzGEOZSoBcfBOIHuGQwkHqlwmBOOf81UUvcT/gZDqmIQGyYjTcz+aWv+Hf8AivWBb7n+qWv+Hf8AivWBRPsqPRNORn7FA3JUBUUDPZ0HAdBLwvHequ+QRP8AVZ66WnY9jnxxNdoiZjVkySPz7P8AVYMnouRxnwo7hK501ufV+sTvoaeqnHh6PBfKwP8AD5nOkEb7e5YrHb4rneaCileyNlTUxRPe8hrWtc8Akk7AAZX0/wBJPCdbUekiu4puzbY7ht1zhEkkNyp5j6qHtZsxry8+wDsApdjR80k4Sv8ABZGX2azXGO0yENbWup3CF2eWH4xg9DyKNHwpfq+01F4pLNcqi202fGq4qZ7oo8c8uAxt17dV9e4hdc+H7vxvxLxBebdLQXWiqLda6WmrI521rJMCHREwnRHGwB2XBuCABut9VK6ycZ03GRvtvZwVa7SILdFT1bD6zGafR6s2AHVrdISX6mgDck8kr0Gj5X/QfxeCbFdqJtdWXe83GopYKOFmsOiia3drQNRcXH3YC83SWqvuFxZbaSknqK2SQxMp42Fz3PGctA77FfcI7zb6fgWk4RslVSUPE1Hw6KqO4tqW/lWyvMtRSMdnEchbjcHUcFvVeB9Cc8MHpBpKh08EM8dNVOpPGlEbXVBhcI26nEAEk7ZISTdgePgs9wqYZZ4aKpkihlbDJIyMlrJHHDWkjkSRgDqundeBOKLDRGuuvD12oKUPERmqaV8bA48gSRsT07r65wJwz9G03CXCNfWUclwu3ErblcIKaqZN4MFPHkCQsJAOSTzXO4lmuPDto48uHEtfSyVvFbxDRUTK1lS57fH8QznQ5wY1rAGtzgnOAFaexHyO12W5X2tZQWqgqq+reCWwU0TpHkDmcDp5r0Fg9F3Fd84uh4XNluFHXP0umFRSvHq8R/tXjH1fPryG673o/sTrxwZxQ+zV7Ib/ACSU9MIn3BtI1tESXSyEuIDxkNaRnYHkV7ifwmcdttNsu9JVT2fg00dBIK5gbW1L4nE6ZC4DYyOIBI2aE29gj4lVcNXWlvzrA63VhugmEDaTwXCZzz9UaOeSCDjzXtYPQ1crTxnUWDiWCuhigttRcBLSwuJnMcGsNjyDq9shhIHPOF4/humjk4ytNJcKsUzDXxRVFQ6XaMB4DnawemOa+jXe8VvDNx9J16kro6a71c7aCjEVUx72QzzF7nRlrjt4UYGRy1DOEKVhR80uvDF8slfBbrnZ7hR11Q1roqaanc2STVsNLcZOTtskvnDl64ZqWU17tVbbZpGeIyOqiMZc3uM899l9p4RvtHSzcOW31ulr7tRcK1UlCH14izW1MxcYhNn8nIIs4OQQ44yCvBeke4VkFNbuF6izUdrhoHy1jI4rh66/VNgHVJrcB+jHsg8zk801bdITPL2zhTiC80E1wttkudbRwEiSenpnyMYQMkEgdBuewXTrvRxxDb+CqDjCeinbbK6SRrX+C7EbGloa97uTQ9xIbnnpPkvdWi2urfRxDS8TNpKC2W2jqKu13ijuUYmZJIdXgSU4fqlL3gNxpDm5+tgLVQW+3z1/oukrKyhksUdEynk8SrYWCs8WeURzR6tTW6/DBLhjB5p0Fo+X1/BfElotkV0uFgutHQS40VM9K+ON2dxhxHXp3TVPBXE1G2ndUcPXeIVUwp4NdJIPGlIyGM29pxHQZX0+KhucPDdXb+Kb5SyXfiriKhpa5j7jHMaeFhdI+R5a4taCSORwAOnJcuPjHiDij0v3biCzCnuBikqGwUtVWtgYaM5hDI3Oc3S7w3bFpBGSQqTEfPL1w9d+HKltLebXW22d7dbY6qExuc3JGRnmMgjI6hNa+Gb5e4JKm12W5V8MJIkkpqV8rWHGcEtBGcb4XX9JtBbLTxCyjtFTUSRMpInyU0tWKoUU7gXPgbK3Z4aTzHUnnzX0mxWv6N4q4avEN7t0PDNntTam3j6QjY6pqRTOkkAjDsh5mLi7UBsMb7BPloKR8ktHCHEV/ax1psV0r2yF4a6mpnvDtONW4GNiQPtSWjhe+X+tlorVZ7hX1MAJlhpqd0j4wDglwA23236r6DT3KWWH0ZcKQXpkNHNIy5VzoqkNYyaWrc4mQ5wCyNjdjyz5ruX+Obj2wXCXhu5W6iFw4mraq8ySVjKcxR5aKZ7skF0YaXkac+0TtlGwpHzPjThaLhJ1lpXSVBrqu1xV1bDMzQaeSQkiMDmMN0k53yViHB/Eb6CW4NsN1NHDEJ31Hqj/AA2xnk/VjGnz5L0fpYrWcQ+ky8er1Ec0LZo7fBP4gc17I2tia/VnBBxnK9ncOJoz6R+Kq6G6tdbOH7FU2+gYJwY5gIm07GsGcO1OcX7ZzjKpXQtHya3cK327UFRcLfZrjWUdLnxqiCme+OLAycuAxsN/JdOh9HHEVy4OreLae3zuttJKyMuELyZAQS6Rpxgsbj2nchkL6XEZaTiHhHiKmvtBR8FcP0VNNG6KsYHBzWAzxGAHW6aSTU05bvnnheTvMpqvRJQz2+ojMD73WVddAKpgfTatLImeFqDiCATsCN0uXsOjwVrs1yvtayhtVBVV9VICWQU0Rke4DmcDp5r0V74ArbJaeHxNSXJl8u0lQHW6SAiRrGP0M0sxqJcc/wCy9DwDTT13o94qoLFU00V9qKmkbI2SrZTvfQgOLw173NGnXjUAc4AXra2oFPXV3D1pvdFNfbdwrS2621YrGBsr3O11IilJ0h5adI3BwCEtpgfGq3hLiG3V7rdV2K6U9YIjOaeSle2TwwMl+kjOkDryWFtrr309NVNoqkwVchhp5RGdM0gxlrD/AFjuNh3X2nhq4w2q+cG8H1l5pnVscFxp62qFQ2WKgbVRkNhMmdLtONRAOATjK20vFHDNPwzPUQVVMaPge55s9M/SJKx3gaGvA5nVNmRx6aQhsKR8apOCOKLhM6Cj4du9RIxz2OZFSPcWuZ9cHA2I6hcZ0bo3ujka5rmktc0jBBHMEdCvr7b3LNdPRtYHXoMpBou9wk9ZDWmplmdK90js41ANxvyyvmXE1z+muI7rcwABWVk04xyw55I/lhCA5RGyGE5CmMJ0FigbIgIlAoAYM3w0581Y2TAA0RnzwqhsdkTnK0i0iWgpU5CVRQ0armPzO1/w7/xXrnEd10rmPzS1/wAO78V655Uz7HHoVTO6J7KALIoIQOkHIaAvc8PeiqqvVNQia92m33G6QuqLfbqkvM1RGGuIedIwxrg06S7njPJG6cCWW2+jOi4ikv8AA+7VtVKIKVrH4kijw1zQeWoOJJJ2xsEcvgKPCDAOQAjgZzgZXdg4MuNRw/QXqm8Koirri62RwRkmUTgNIBHZ2oYXpZfRFPb5ni7cSWK3QGqdQU9RK97m1U7cCQMDRnQxx0uedge6afyB88LAeiPh56bL1V+9H934btlzrbp6vA63XP6Kkg16nvm0ayW42LQ3Bz5hdin9EVwdPcW1l5s9vgtlLSVdXU1L3COJlRgtbsCS8A/V68gio9hbPMcKcT1nB1zdcrdHAak081M0ytyGCRhYXD+8AThcUvAwBjYY+C9d6PeF7XxP6QaSy19xibbBM9z5iHM9ZijySGDmC5oyM8l3rtwWOL7my4x3Dhey2yon+jrS6mgfFHXPBGA1gBcdOprXSO6kc0m66D+z5nqBG+EdIcMYGF7W0eia6V07oLjcLdZ5H1sttpW1Je91VUxu0vawMB9kOwC87ZIWir9HVBZ+Aa68XS800N4iuL6JlGNbsOi1CSPI2LydJzyA8yEcvkKPBeF2GyHhAHlhei4UreHKRtWOIKCWs8cwxR+HzgZ4gdLI0/t6QWgf3s9FzLpUU9XcaqopKVlJTyzPfFTszpiYXHS0Z7DAWnFE2YgwAEYGEzSG8gAoScqbdE+ugD7Oc6RnugSMk4GTzU67qItiJpbj6o38lCwHmAUQiigIGDG2B5JmwtcfqjKYN5eadrcFWoisUMAyMDfmnODzaCmcCTnAHuR0bfatEiSo8sckWkbZGw/kmeB0BGOe/VL0x0S6AVzGuOotGrvhJobnOBnuremMBAjy9yVWMrIB5gH3qaQRpwMduifG+6mnr0SodigNA04GOyJxzwNkceSGE6EIWjfYFHSmwiG7JUMrxtyQxjYqwhAhFBYhHdDCsx0QwNuqXELEx7ij9iYjOUNk6ABQyiSpkYGyQzVcf1S2fw7vxXrAt9wH5pbf8h34jlhUzWwQNOx8kC3II7ghNvyQWbRR9cb6QODae9f0oiF3+lJbMbfDStgaIqGUU/hatWfbB3xjkDuvIcS8Q2i8cE8MUNNJVwV9nhkpZaV0QMT9by90zZM8zkAg9l5NAhFfAWfWOD+IZPR/6MauvuFA0V1XXMquHvFO4l8IxvqdP7LWnYnm4jsuXR8XcK3Wz8OHiH6WFXw810Xq1LG10dwjMplbl5P5N2o4cd8jcbr568uk063OdpaGjUScDsM8h5IBuEuLCz6bV8e8O8XWoHipt0hqmXqouk1PQRtcysbKGjQHuPsaQ3Tk9PNZuNPSPRX+38Q01FDUxvu96irMuaGtbSxRaIo++Qf9F88UPPZUopCs916PIHWKivPGVViKCkoZ6Wie5wzNWSt0Na0czpDi49gF3bB6TKOk4KtVpF9vliq7RHNHpoKZkrKzW7U1we7eN/QnthfKck48uXkiChRQWfVLZ6UYRwXa7ceI79ZKy1RzMdFQQNkFaXvLw/xDux2SQSewK4NTxBY73wbw9a6+tuNPVW6uqH1QZB4olZO9pdKHZ3eA3GDzXieaOnbKfFBZdXNpW1tQKF0z6QSuEDpmhsjo8+yXAbBxGMgKhHCKqhCny5KYTYUwigsGFMI43RxndOhC43TAb7IgIgdE0gHAOBlWMYSCeeEgCZoPwWsSWbIKWSpy2KJ8jgCcNGTgc1Q8hrcdVfR1c1LIJIJXxPAI1MOCB1VEjgXbcvNayriqJ3ZUe3dBzcHlhWcypp1bnKyoor0jChCt0F2B2RDRjkcp0BRpKOjZXaMtVYc0lzQ4Et2IB5Ipe4ClmyBariAc4OeyAZnyT4hZVp/+lMc9lZpwlIGUqAQDrnkhjKbryUISoBCFAN90XOaDu4fFIJo3PDA4EnslaQwkbIY8k+EC3dVQilRHbG2UM9FiaGyvH5pbP8h34jljc0c2k4zyPNbq39Ttv8O78RyxPOTnYe5VJIlMrwjhN9iIx1WdDEx1UwmwoR1SoYAN8KYwrXUs7ImzPgmbE/6sjmENd7jyKTSeSdCFwjpTJ3wSxMikkiexkzdUbnNIDxnGR3GUAUlu6OE2N9uSIYU0rAARwmMTmsDy0hpOA7GxPbKDmuY7DmuaeeCMFOhExtsh0HkmwrI6aWSJ8rYnujjxreG5DMnAyemSnQFIGT2RwrTTyCMSmN4jcS1ryPZJHMA9cZCXSAOe6KAXHTCOlEBMG5O+yaQmxQBjqmA2TaUcc9lVCBthTBCYDyTNbnA5JpAPG4NBJznG2OWfNIRkpgMKYIyM7HmroQoCtA9nHbdKBjHUK1o2PRNIBdOyLYcqwM8lqpYHSOGlmrG526LRRtisxOjOeS4bqeWSscYQdLiSDnGQvV10LYmaQCHP2AHTufsCw1NMWU0giYC5jW4GOvNZZ8HJb9iozozx0pDQ5sjsdiVNJycudnttlU0c8o1OlyA86+WBjuP91tcwPGDunDjJWhO09mZ7Xftn4JHNeDgkfBXugfzHtjsef/KrkZ+TcdeNt88whxBMoPiE7BqqkkbFgzDn2dn+SpkryGARj2sbu6fYsjnOeS5xJJ6lcWXyEtRNo437ls1U6TZo0N8uZUo2652DJGN9lQulb4MDWME4/wBVhhUsuRWXOoxNOFNKctzspoXrcTlMfZDCIORjG/dDOSuRmxurv1K2fw7vxXrEQuhXNHqFrP8A8d/4r1hxhXJEIXTgBQjbZMjjyUUM6LbXaCxpdxAxriAS31KU4PbKzVUFNTTRtpattc0gEnwXMGc/Vwdzn/dZ9PdQZBBBII3BHRAWfQ33R/EdbFeGT3MwwXGkbVWmvc51M3U4NaI8EDbB9ktBA7qo0FjnqLzdrjQ0NMI7oKIU7xU+GASS54EZLtZHLJDR26Lxtderpc444665VtUyN2pjZpnPDT3GTz81dFxHeoKl1XHd69lQ9rWOlE7tTmjkCc7gdFQWerprNw74top4LRLXw3OuqIPWpJZGSMia8Bha0YAcAcnUDsOSu9Qt1X/R6xVNEakyUFQ2OrbK5rmBr5C0saNjkt3znIPReXHFt2ZbIaCCsqadrHTPkfHO4GcyEE6xnf3+a5ra6rjkheypnY+BuiJzXkGNvZvYbnkm6EeuqbNw7QWuGnldG6rmt8NXHOxs7pnSvwcYx4Yi6dxzz0Qv9BZGw8Q09DaxSvtMsAin8d73yBzmtcH5265GAMea8s273JtB9HCvqhR8/V/FPhjfPLlz3VbqyqkNRrqZn+skGfU8nxSDkau+ChMD0LXWt/A9Ay5y1zGi4VJYKVjHf1G5J1ELvVNpoqriSpqa+lp6qhkloqOJ9UZzKS6Bp0tbD/W0jOTsPNeA8WV8DYHSPdCxxe2Mn2Wk8yB5rXBfLrSumdBc62J04aJSyZwLw0YbnB3wBgdlWgs9jBwXY2yS1FQ6VlDSTz22ZxedXrJmLIXHyDSHEDnoK023hq308dxsFZMKSWKKhFcde8s5mJ8NpOzSRoAJ2GclfPXVtVI2Rr6mZzZZBK8F5w945OPc7nfmjLVVU7pnzVM8jp3B8pc8kyEci7uR5p8l7IR66kttJPBTmvtktO1k9xfJQCZ7Wx+FA1zWb5I3G55lVw2S11NsF7jteXC2vqfUGSyFj5RO2LVnOvSGu1EZ6c8Lzf0hWvJL6ypcSXEkyEklw0uP2gYPcLZars+31EUkprJGQscyIQ1ToHRauelw5eY6p9isfiigpqKro/VqJ1E2ahgnfCXudh7gSTl2642nyXTv15lv1wNXIzwwI2RMZrLy1jRgZcd3HqSeZK54aShoYoamDUzQMHbfujhFCAG775R04Pf3JgNk2E6ArIwjp2KfTko6VSQiRsDiAThXNgIBJB2OFWGZydvctkMrgNLvabjGD0WsUhFTWY5jK20UroCS0/WBafMFL4LDGCCMk/BLMHRjDN3uOlmO5/8A7K0X07Dskr5J5HaS1zP0bBjcEbucP9FPBJfLtv4h/krqanayq0gDRCxrN9tzvlXUsQlh1nfU5zviUKN9is4poAHywA6dJ8SN3YH/AJ2VTWOgw1+wzpx+ye3uPRdW8PioBBUveIwCWHHMg9QOuCvKXG6+syu8Br42kYLnOy546ZXLnnDD/ZpBOR0au5QUgLc+JJ+y3p7yuHVVk1W7VI7bo0cgqTugvLy+TPJp9HRGCQUEU0UL6iQRxt1OKwSbdIsamh8Z/tahG3dzgM4Xbgh8GMNxgncqiip2CNsYYQ47vOTuMroFuV7HiePwVvs5Ms7dFQbrcASB0z2R0AbZz5pwxN4Z/ZPwXaomVnGU6qYyd9lBscc/NeYdJ0q79QtX8O/8V6xYW+rx9H2vv6u/8V6xELWSM0LhFTHwRwoodkwNPXKGE2NlMJgLpUwn0hFrWk4c7RtzxlFCExhHGcZTYyBtuoBunQAAHZHCICZo78k6AUI4ThqYs6/YnQCacf8A1hO1vdEDKYBNIVihqJCcNynDNWBsMK1ERUG75wEfDVwjzsnwCSdIB8uifEDOWYR07K7w++VPD/4RQFQamDfJWiNMIx5p0BXpyjo/mrWxjtunEapIRS2NXMZg7J2x+SubFndWogCI6cggHIwmo3Nmn8bI0My1mevd3+yqqg4ARx7OfkZ/ZHUrNXVMNup2a3AAA6W8ydtlTnx2+kCVnVpZB4Es7wDrc+QjG2By/wBFyq3iint9KyClDaicN3d/UYf9yvP1t8qKuEU7D4UAGC0Hd3vK5xK8/N53tj/02jh+S2rrKiumM1TK6R56np7uypUKC8yUm3bOhIKiVdK22n12MyyTBjNwANyT59gjHCWSXGK2KUlFWzHBTPqCdOzRzceQXWjhjo4GNjLdbvrO7jqP9lc6SNtFFAWNaQwE6OR8/erqOnZVPimDfZjHMjGXbf6L1MHjKGo7bOeeRvvoeCmMTAHAaz9byPb7FaIz2Wl8Qa4tyD5jkkLN16igkqRzWUhvkrDLJnZ7wPMo6MHAR8PPUBNaEea9ygyoeiYHovGOw6VZ+oWv+Hd+I5ZMLZVfqNt/yHfiOWQraRkDGARgb9eyiOEcKRgxsjhHCmEwAN1MbpgOiYBAAAPJQN7pw1NpyqoRXjCICfSjjdOgABjknAGFAE7W7c00gFDU7W45JmtVgaeeFSQhGMHI5A8k4bgJw1MGeStIRPDa4MDCdZ552HlhAMz0VjW77J2s2IBxnmqqwKdO6IGMjCvEaJhIJGMIoCjRnonbHgq9sXkrGxZCKAobHurBFkq9sXTCtbD5KkgM7YVZhrGlziA0DJPYLQ2E9lwOK4rk5jWwRvNJpzIY9yT5+SWWfpxcqscY26MVdxEyMv8AV2h8zjjUfqsaOQ8z1XAnnlqZTLM90jzzJKTGOeR7wiGOcfZBd7hleLkzSydnTGKj0Lt02QO3RaYbfWVDtEVLO889mFbKbh2oc/8APHeqMEgjIc0ufv1DRzCzjCUv2oq0uzkrVSWqsrmPkhhJYwZLicD+a9FS2azwmWPxvWi9zmNMkTg4dBgDkc91jra6S1yNjbKZXCPS0uaAQAdg4cu62/TKK5ZHr+CfUvUTLQ0VNK59M9ntPaR4j9nRuHl0VNPXG3QywsaDLrOHjduOR9/kqblVNrKkzMZI0uHta3ZJKy5UPMov/j1XuHG+zZSuqKyZwDtWhhJbnGQOgXc4ffK+KRr4jGwO1Mz2K49mMUdygL3gtfkHHQnuu1bq0RS0tJp1+K90evP1cHZd3hOqnKXz/wCGWXekjqmMHoo6InotfhDG/NDw17By0YnRckujyW4w+STwvJKgPFItBJUxg7ojGd8rxUjsOnUj8wtv+Q78Ryy4WuoH/T7Z/kP/ABXrNhbyMhcI4RwiAoGABMGogJgEwFAyiG7pg1OGpoQgCcDZEBOAqEV6EQxWhqYN32TArDE7WKxrMqwRpoBGx91Y2NO2NWtjVIRSI04Yr2xeSsbHkKgKGxp/DPZaGwq5sOQAnY6MjYj2VjYT2WxlOR0VrICmOjEIDscK1kHcbLY2m8lcylI3wmOjC2DphWiE9ltFNhWNp07DiYREQnEJC6DabIIwN05pmMGXODfeU+Q+Jy/UIJcCWnidg5GWDZM3h+3yEH1OMEci3LSPgupBDHO7TG9rj5FbGBtC7W5zGkA/WwpbiVGDPNycO0mXZExB2I8Z2EDYonH9JVdOU2OXLouqa2le8gSgEd0grqbVgO1Y69EVEHFnMHDlF4he6Fz3HnreTlcfiaktVumt3rNNE2B0zi8tZ0DevUhetN4t7WOy46xy7LzXE9PTX+KKMzuiMTy5rmtznIxhRk/a1BKxKO9nkeKH2p0sTLbFHkDL5Ij7Ls9MdwuCRutVfBFBUyRwSPliYdIkcMaiOaz4XjZ25zbar+jeKpGmntle+L1qGnl8Ng1a8f6d1psviG70hcXH8sHH39U1oucoqoYqmolFMXNad9hjkM9B7l7DxGsJ0xsae4AXT4+GEqlF9ESb6Ztc+Jo3cM9lmkrGD6oHuPdZ3Pe5VuY49CvVeVvow4IZ9Q/k5+x/ZQ9bA23+1Vuid2S+E7so9RoOB5g8kPrHbZNgkHn5pcYGTyXE0anXm3t1sP8Agv8AxXLPhaZR/wBNtf8AkP8AxXqnC0ZkJpRATYRAUjAAmAUwnATEABMGogKwNTsKFDU7WohitaxOwoQMyrGxpwxWNYmOitsataxWNYrmMCpNBRU2MlWtiVzGjqB8Vexrew+KOQ+JSyBXMp/JXMLfJXtI7I5DUChlN5LQym5bKwF3QtHvKsaJj/aRj7UOZaxkZS+SuFIFGNqCNpokzqComHtTRn/8lPqlrEO2Bo54VoiaWgZyB0ys4s8xGPHY0faVZHY5j9aob9gKn1S1hYJGyhw0U5cP+4KqaeSE6fAcCunFZ52t9iVn2hWGzV8owZodPuUvMaLCcUS12gyANDR3XPrKyaowJcezyAGF7Okss9PnMkTgee2Uai1Usx/KxgkfsR4Wf6jZp+n0eIp5ZYnamOLT5FX1UrKjB8PQ7HtYcTn4r1Bs1HghlLL78Kl9gjPKKoPkGp+sg9FpUeQfEz+8k8EZyG/Er1T7CAdqKbH95yYWnTt6m0e8p+qjN4WeSMDncmj7FzL5O+hp2xsIFRUHw4wemdi73BfQ3UTYGOkkZFFGwFznY5AcyvJUFIy9eucQXFkcdFU5pqZsg2bFyyBz1HpjdEsmqRDx0eMvlI2hoKLPtxhz2jHIkDn9pXng/K+h3uKpiqbXR1FIYmRiYwyzAN8eMMOCWH6p8j71865Lz/Im+VoXGtHo6OjE1pfJIwsax4bI0f1oyMh3/c3nnsuzYnvka+iqCH1EAGHfvI+jgjwxUS3GUUsbGxsEMMxdO0nOnLTpHUHK7d14a9Wp47pZA71ihJcYDvqZzLB7t8BdeKSSUkLhor9Uf+wfglNM/thd61yUt5oIq2md7Eg5fsnqD7lY6gbqwCD9q6vUQ/TPNSwCGN0srg1jeZK5Rv1tBIc5wIJBBHIhdbjaN1LQM1MD6aTZ2kkPY4cj5hfNnPYXHLsHyasMvkuL0Q47o3OPtDOnfGzVHHIA80EW4yMnC07MjXcKmSno7WG6SPV3Hcf4r1gN1m/ZZyxyWm8OHqtrHandn7165RXLmySjKkyoRTWzr0tdHPpactedsdFsxhedjOk5BxjdXsqpW7iRwPvVQz62KUPg7eQmacrhyVU7wcyHGOQ2Tw1c8eD4hcOzt1ay26FxO81WsAI239y87LUSTn23kjt0SMc5hy1xae4Kr1BUeoaFaxuei86y51bG48QHzIyUhqp3nJnkz/3FP1APUgJgvPQ3iqixlwkHZ4/3W5l/ZjLqc58nbJqdjtHWbqVrGE9VyI+IG5GuncB3DsrWziCiBAd4rfMtT5DTR02Qkq9tOSsMN7t7nYFUz7chWzcSWqj0+JVscXODcM9rHmfJS8qW2aKKZubTHurG06zW/iO1V0HiNqo4iM5ZI4BwAPNPeb3S2q2mtaWzatogw5Dnb437bKHmVWWoI2R0+TzWuKkJ5L5dUcc3mWZskc7KcMJIjjbsfI55pG8b8QCIRNuMoDcHIA1bHPNYPy0UqPsMVCRjZa4qMr57wn6Sa+qudNQ3KOOdk7xGJGNDXNJPM45hfWInUscXiyVELIwcFzpABntlQ/J+DoxwUtmWK355nC2RW6Pq5ysoLjaa4QmluVHMZ8iINlGZMc8Dnsum40tHp9ZqIIATgeLIG592SsJ+SzpjiTVoyw29g/aK2R0Lf3ZK69vgp6lodC9kw7xuDv8ARdqnoWY+rjHPIxhcsvMdm8cNI8syjcBtAfgi+mlA/Vz8F3pb7wzS3FtuqL9a4KtzdQikqGg49/L+a+c+kv0ucPs4VroOGL2yS5vLY43sBDmjVhxbnrgc/NKGac3SRMpQgrbPTMoqmQ4EH2A7qPt8xGDGc+9flqu4uv1wqIZ6q8XCaWna5kchmOpjTnIB819D4K9N9TaLdNDxE+ruj2uY2nDQ0PDMe1l559OeVu/UXRzx8nHJ09H1iW2zfsH4hY32uXOdGPtXgL3/AOoSR9VCLNZminaPyvrrvbf5N08v5rl1fp9vUrHthtNthJ+q4l79O/UE77Jxll+BvPhWrNnpJq7tU3qj4Rs7mGavhzMzYE5JIGo/VGkZK4tpHGt0LJrULa6CizTwTaGtiGNiI9Y383AfavF3bia5Xa/VN6km8GrqCcmEloYCMaW9hjZdGP0i8TU9HDR0lzNHTwRiNkdNG1mAPMDJPmnc2cjnFyth4yq+JBdBQXx8ZqKdhc0QtZgNcMk5aN8heZwFouFyrLpUmqrqqaqncADJK4ucQOQys2pVyrsylt2dG11NWypHgXD1QiMt8R8haA0b6f8Ahd+GSGmqmi6cUyzQzxF+aOd+ztvZfkbbHt0XjslDKPUroEestfFUfD09xpKGofLRzNJgkkZu2TGxx/L+aWg4rgoogwCpdJ7TxKAAWPIHTrvleVyoOeyazyQHq73xlDeKZ8Mlv3LcNkMh1NOOe3TO+F59lw0NDTTU7yP6zmblZPeopllk9hXydfy7JX7KKL2JdHKiXAl1DQuPNviMHu1Z/wByuaVFF5/kfu/z/o1h0LlAOI5FRRcjNBvEdjGVY2V2nGyii1xyZLQ3iHZNrIIUUW9uiCwOOQnyootYkMSWRzGZbsdkGTOGeu/VRRS2+Q60CWqkaW4ICVlVLpdkgnnk9FFFm5Pn2UkuJUZ5NWrWcqtzi8lx5lRRcrba2a0LnBV/rtS6lFIZnmAO1CPOwKiihFFXNBRRUIZjnMeHMcWuByCDghXPmlLNJlkLc6tJccZ7qKLSAmBkr2kOa4tcNwQcEe5WyVM82DLNJIRy1uLsfFRRbLohm6juFZSPa+mq6iBw/rRyFp/kurLxrxPNSyUknEV1fTytLHxGpdpc3sfJRRacU+0QpyWkzhEA5yAffuq3e4KKJiKyBnkjyCiikYhcVW5xUUWUi4iZRztlRRYosUnKiiihjRMqKKJARQc1FELsCKKKKgP/2Q=='

with st.sidebar:
    st.markdown("""<div class="mc-brand-lockup"><div class="mc-planet">◉</div><div class="mc-brand-name">SpectraAI</div><div class="mc-brand-sub">EARTH OBSERVATION<br>INTELLIGENCE</div></div>""", unsafe_allow_html=True)
    st.markdown('<div class="mc-nav-label">Mission navigation</div>', unsafe_allow_html=True)
    selected_nav = st.radio(
        "Mission workspaces",
        ["⌂  Mission Overview", "◇  Observations", "▥  Spectral Lab", "⬡  Quality Control", "◎  Anomaly Lab", "▧  Image Analysis"],
        label_visibility="collapsed", key="mission_navigation"
    )
    # Each navigation item opens its own working workspace.
    if selected_nav.startswith("⌂"):
        active_page = "Mission Overview"
    elif selected_nav.startswith("◇"):
        active_page = "Observations"
    elif selected_nav.startswith("▥"):
        active_page = "Spectral Lab"
    elif selected_nav.startswith("⬡"):
        active_page = "Quality Control"
    elif selected_nav.startswith("◎"):
        active_page = "Anomaly Lab"
    else:
        active_page = "Image Analysis"
    st.markdown("""<div class="mc-sidebar-footer">EXPLORE &nbsp;·&nbsp; ANALYSE &nbsp;·&nbsp; DISCOVER</div>""", unsafe_allow_html=True)

# Pick a fresh field note whenever the user returns to the Mission Overview.
previous_page = st.session_state.get("_last_active_page")
if active_page == "Mission Overview" and previous_page != "Mission Overview":
    st.session_state.pop("mission_quote", None)
st.session_state["_last_active_page"] = active_page

if active_page == "Mission Overview":
    try:
        overview_data = load_data()
        total_obs = len(overview_data)
        areas_count = overview_data["area_id"].nunique() if "area_id" in overview_data else 0
    except Exception:
        overview_data = pd.DataFrame()
        total_obs, areas_count = 0, 0

    # Use the inventory itself for the file count and the known HLS layer stack.
    tiff_count = len(list((BASE_DIR / "data").rglob("*.tif"))) + len(list((BASE_DIR / "data").rglob("*.tiff")))
    layer_count = 5
    obs_label = f"{total_obs:,}" if total_obs else "—"
    region_label = str(areas_count) if areas_count else "—"
    tiff_label = f"{tiff_count:,}" if tiff_count else "—"
    dataset_state = "Dataset Loaded" if total_obs else "Dataset Not Loaded"
    analysis_state = "Analysis Ready" if DATA_FILE.exists() else "Awaiting Dataset"
    session_state = "Local Session Active"

    st.markdown(f"""
    <div class="mc-topline">
      <div class="mc-dataset-pill"><span class="mc-pulse"></span> NASA HLS Dataset &nbsp; · &nbsp; Sentinel-2</div>
      <div class="mc-top-right">◉ &nbsp; SpectraAI v1.0 &nbsp; | &nbsp; ◐</div>
    </div>
    <div class="mc-hero">
      <div>
        <div class="mc-welcome-small">Welcome to</div>
        <div class="mc-welcome-big">Spectra<span>AI</span></div>
        <div class="mc-tagline">ANALYSE &nbsp; • &nbsp; VISUALISE &nbsp; • &nbsp; UNDERSTAND</div>
        <div class="mc-hero-rule"></div>
        <div class="mc-hero-copy">An advanced Earth observation platform for multispectral analysis, quality assessment and anomaly detection using NASA’s HLS data.</div>
      </div>
      <div class="mc-status-card">
        <div class="mc-status-title"><span class="mc-pulse"></span> &nbsp; System Status</div>
        <div class="mc-status-row"><div class="mc-status-icon">▤</div><div class="mc-status-label"><b>{dataset_state}</b>{obs_label} observations</div></div>
        <div class="mc-status-row"><div class="mc-status-icon">✓</div><div class="mc-status-label"><b>{analysis_state}</b>Local analysis modules</div></div>
        <div class="mc-status-row"><div class="mc-status-icon">◷</div><div class="mc-status-label"><b>{session_state}</b>Data processed on this device</div></div>
      </div>
    </div>
    <div class="mc-panel mc-mission">
      <div>
        <div class="mc-section-title"><span>✧</span>Mission Overview</div>
        <img class="mc-mission-image" src="__MISSION_IMG__">
      </div>
      <div class="mc-mission-copy">SpectraAI empowers you to explore satellite imagery with scientific precision. From data validation to anomaly screening, discover insights hidden in our planet’s spectral data.<br><span class="mc-explore">Explore the Modules &nbsp; →</span></div>
      <div class="mc-metrics">
        <div class="mc-metric"><div class="mc-metric-icon">▧</div><div class="mc-metric-value">{obs_label}</div><div class="mc-metric-label">Observations</div></div>
        <div class="mc-metric"><div class="mc-metric-icon">◎</div><div class="mc-metric-value">{region_label}</div><div class="mc-metric-label">Regions</div></div>
        <div class="mc-metric"><div class="mc-metric-icon" style="color:#b58aff">▱</div><div class="mc-metric-value">{layer_count}</div><div class="mc-metric-label">Spectral Layers</div></div>
      </div>
    </div>
    <div class="mc-section-title" style="margin-top:22px"><span>♧</span>Quick Access</div>
    <div class="mc-footnote">Research prototype · Candidate anomalies require human inspection and are not confirmed real-world events.</div>
    """.replace('__MISSION_IMG__', MISSION_IMG), unsafe_allow_html=True)

    # Functional quick-access tiles: clicking one updates the sidebar workspace.
    quick_col, quote_col = st.columns([3.0, 1.0], gap="large")
    with quick_col:
        with st.container(key="quick_access_cards"):
            quick_rows = [
                [("Observations", "Explore validated satellite data"),
                 ("Spectral Lab", "Analyse spectral indices & patterns"),
                 ("Quality Control", "Assess image quality & coverage")],
                [("Anomaly Lab", "Screen unusual multispectral patterns"),
                 ("Image Analysis", "Upload & analyse your own images"),
                 (None, "")],
            ]
            for row_index, row_items in enumerate(quick_rows):
                tile_cols = st.columns(3, gap="medium")
                for tile_col, (page, description) in zip(tile_cols, row_items):
                    with tile_col:
                        if page:
                            tile_key = page.lower().replace(" ", "_")
                            tile_images = {
                                "Observations": OBS_IMG,
                                "Spectral Lab": SPECTRAL_IMG,
                                "Quality Control": QUALITY_IMG,
                                "Anomaly Lab": ANOMALY_IMG,
                                "Image Analysis": SPIRAL_IMG,
                            }
                            tile_image = tile_images[page]
                            st.markdown(
                                f"<style>.st-key-quick_tile_{tile_key} [data-testid='stButton']>button{{background-image:radial-gradient(ellipse at 94% 94%,rgba(3,12,30,.99) 0%,rgba(3,12,30,.96) 17%,rgba(3,12,30,0) 43%),linear-gradient(90deg,rgba(3,11,29,.96) 0%,rgba(3,12,30,.82) 58%,rgba(3,12,30,.58) 100%),url('{tile_image}')!important;background-size:cover!important;background-position:center!important;}}</style>",
                                unsafe_allow_html=True,
                            )
                            with st.container(key=f"quick_tile_{tile_key}"):
                                st.button(
                                    f"{page}\n{description}",
                                    key=f"quick_{tile_key}",
                                    on_click=navigate_to,
                                    args=(page,),
                                )
                        else:
                            st.empty()
    with quote_col:
        # Original rotating field notes avoid questionable or misattributed quotations.
        import random
        rotating_quotes = [
            ("Every pixel holds a piece of the planet’s story.", "SpectraAI Field Notes"),
            ("Look closer. The Earth is always telling us something.", "SpectraAI Field Notes"),
            ("From light, we learn to read the landscapes below.", "SpectraAI Field Notes"),
            ("A clearer view begins with curious questions.", "SpectraAI Field Notes"),
            ("The smallest spectral shift can reveal a bigger pattern.", "SpectraAI Field Notes"),
            ("Explore beyond what the eye can see.", "SpectraAI Field Notes"),
            ("Our planet is vast. Every observation brings it closer.", "SpectraAI Field Notes"),
        ]
        if "mission_quote" not in st.session_state:
            st.session_state["mission_quote"] = random.choice(rotating_quotes)
        quote_text, quote_author = st.session_state["mission_quote"]
        st.markdown(
            f'<div class="mc-quote"><div class="mc-quote-content"><div class="mc-quote-star">✧</div><div class="mc-quote-text">“{quote_text}”</div><div class="mc-quote-author">— {quote_author}</div></div></div>',
            unsafe_allow_html=True,
        )

# ============================================================
# DEDICATED DATA WORKSPACES
# ============================================================

if active_page in ("Observations", "Spectral Lab", "Quality Control"):
    try:
        workspace_data = load_data()
    except Exception as error:
        workspace_data = pd.DataFrame()
        st.error(f"Unable to load the NASA HLS analysis dataset: {error}")

    page_kickers = {
        "Observations": "MISSION DATA / OBSERVATION EXPLORER",
        "Spectral Lab": "SPECTRAL INTELLIGENCE / BAND & INDEX ANALYSIS",
        "Quality Control": "DATA INTEGRITY / PIXEL & ATMOSPHERIC QUALITY",
    }
    st.markdown(f'<div class="mc-eyebrow">{page_kickers[active_page]}</div>', unsafe_allow_html=True)
    st.title(active_page)

    if workspace_data.empty:
        st.warning("The analysis table is not available yet. Check data/metadata/quality_assessment.csv.")
    else:
        filtered_workspace = workspace_data.copy()
        with st.sidebar:
            st.markdown("<div class='mc-section'>Workspace filters</div>", unsafe_allow_html=True)
            if "area_id" in filtered_workspace:
                workspace_areas = sorted(filtered_workspace["area_id"].dropna().unique().tolist())
                chosen_areas = st.multiselect("Study regions", workspace_areas, default=workspace_areas, key=f"{active_page}_areas")
                filtered_workspace = filtered_workspace[filtered_workspace["area_id"].isin(chosen_areas)]
            if "tile" in filtered_workspace:
                workspace_tiles = sorted(filtered_workspace["tile"].dropna().unique().tolist())
                chosen_tiles = st.multiselect("HLS tiles", workspace_tiles, default=workspace_tiles, key=f"{active_page}_tiles")
                filtered_workspace = filtered_workspace[filtered_workspace["tile"].isin(chosen_tiles)]

        if filtered_workspace.empty:
            st.info("No observations match the selected workspace filters.")
        elif active_page == "Observations":
            st.caption("Browse the validated NASA HLS observations and inspect individual acquisition records.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Observations", len(filtered_workspace))
            c2.metric("Study regions", filtered_workspace["area_id"].nunique() if "area_id" in filtered_workspace else "—")
            c3.metric("HLS tiles", filtered_workspace["tile"].nunique() if "tile" in filtered_workspace else "—")
            preferred = ["date_doy", "area_id", "tile", "quality_score", "quality_category", "clear_pixels_pct", "clear_reflectance_coverage_pct"]
            shown = [column for column in preferred if column in filtered_workspace.columns]
            st.subheader("Observation catalogue")
            st.dataframe(filtered_workspace[shown].reset_index(drop=True), use_container_width=True, hide_index=True)
            if "quality_score" in filtered_workspace.columns:
                options = filtered_workspace.index.tolist()
                selected_index = st.selectbox("Inspect an observation", options, format_func=lambda idx: " | ".join(str(filtered_workspace.loc[idx, c]) for c in ["date_doy", "area_id", "tile"] if c in filtered_workspace.columns), key="observations_inspector")
                record = filtered_workspace.loc[selected_index]
                st.subheader("Selected observation")
                a, b, c = st.columns(3)
                a.metric("Quality score", f"{record['quality_score']:.2f}/100")
                b.metric("Quality category", str(record.get("quality_category", "—")))
                coverage = record.get("clear_reflectance_coverage_pct", record.get("clear_pixels_pct", np.nan))
                c.metric("Clear coverage", f"{float(coverage):.2f}%" if pd.notna(coverage) else "N/A")
                detail_fields = [("Cloud", "cloud_pct"), ("Cloud shadow", "cloud_shadow_pct"), ("High aerosol QA flag", "high_aerosol_pct"), ("NDVI", "clear_NDVI_mean"), ("NDWI", "clear_NDWI_mean"), ("EVI", "clear_EVI_mean")]
                details = [{"Indicator": label, "Value": f"{float(record[field]):.3f}" if pd.notna(record[field]) else "N/A"} for label, field in detail_fields if field in record.index]
                if details:
                    st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)
            st.download_button("Download observation catalogue", filtered_workspace.to_csv(index=False).encode("utf-8"), "spectraai_observations.csv", "text/csv", key="observations_export")

        elif active_page == "Spectral Lab":
            st.caption("Explore surface reflectance across the four HLS spectral bands and compare vegetation and water-related indices.")
            band_fields = [("Blue · B02", "clear_B02_mean"), ("Green · B03", "clear_B03_mean"), ("Red · B04", "clear_B04_mean"), ("NIR · B8A", "clear_B8A_mean")]
            available_bands = [(label, field) for label, field in band_fields if field in filtered_workspace.columns and filtered_workspace[field].notna().any()]
            index_fields = [("NDVI", "clear_NDVI_mean"), ("NDWI", "clear_NDWI_mean"), ("EVI", "clear_EVI_mean")]
            available_indices = [(label, field) for label, field in index_fields if field in filtered_workspace.columns and filtered_workspace[field].notna().any()]
            if available_bands:
                st.subheader("Mean surface reflectance")
                band_means = {label: filtered_workspace[field].mean() for label, field in available_bands}
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.bar(list(band_means.keys()), list(band_means.values()))
                ax.set_ylabel("Mean reflectance")
                ax.set_title("Clear-pixel reflectance by spectral band")
                fig.tight_layout(); render_chart(fig); plt.close(fig)
                cols = st.columns(len(available_bands))
                for col, (label, field) in zip(cols, available_bands):
                    col.metric(label, f"{filtered_workspace[field].mean():.4f}")
            else:
                st.info("Band reflectance columns are not present in the current metadata table.")
            if available_indices:
                st.subheader("Spectral index distributions")
                cols = st.columns(len(available_indices))
                for col, (label, field) in zip(cols, available_indices):
                    values = filtered_workspace[field].dropna()
                    col.metric(f"Mean {label}", f"{values.mean():.3f}")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.hist(values, bins=10, edgecolor="#a8c8e8")
                    ax.set_title(f"{label} distribution")
                    ax.set_xlabel(label); ax.set_ylabel("Observations")
                    fig.tight_layout(); render_chart(fig, col); plt.close(fig)
                st.caption("NDVI, NDWI and EVI are spectral indicators; interpret them in the context of the study area, season and image quality.")
            else:
                st.info("NDVI, NDWI and EVI summary columns are not present in the current metadata table.")
            export_cols = [field for _, field in available_bands + available_indices]
            if export_cols:
                st.download_button("Export spectral summary data", filtered_workspace[[c for c in ["date_doy", "area_id", "tile"] if c in filtered_workspace] + export_cols].to_csv(index=False).encode("utf-8"), "spectraai_spectral_summary.csv", "text/csv", key="spectral_export")

        elif active_page == "Quality Control":
            st.caption("Assess observation suitability using the existing quality score, clear-pixel coverage and HLS Fmask-derived indicators.")
            metric_cols = st.columns(4)
            if "quality_score" in filtered_workspace:
                metric_cols[0].metric("Mean quality", f"{filtered_workspace['quality_score'].mean():.1f}/100")
            if "clear_reflectance_coverage_pct" in filtered_workspace:
                metric_cols[1].metric("Mean clear coverage", f"{filtered_workspace['clear_reflectance_coverage_pct'].mean():.1f}%")
            elif "clear_pixels_pct" in filtered_workspace:
                metric_cols[1].metric("Mean clear coverage", f"{filtered_workspace['clear_pixels_pct'].mean():.1f}%")
            if "cloud_pct" in filtered_workspace:
                metric_cols[2].metric("Mean cloud flags", f"{filtered_workspace['cloud_pct'].mean():.1f}%")
            if "high_aerosol_pct" in filtered_workspace:
                metric_cols[3].metric("Mean high-aerosol QA", f"{filtered_workspace['high_aerosol_pct'].mean():.1f}%")
            left, right = st.columns(2)
            if "quality_category" in filtered_workspace:
                with left:
                    st.subheader("Quality category distribution")
                    category_counts = filtered_workspace["quality_category"].value_counts().reindex(["Excellent", "Good", "Fair", "Poor"], fill_value=0)
                    fig, ax = plt.subplots(figsize=(6, 4)); category_counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Category"); ax.set_ylabel("Observations"); ax.tick_params(axis="x", rotation=0)
                    fig.tight_layout(); render_chart(fig); plt.close(fig)
            if "quality_score" in filtered_workspace:
                with right:
                    st.subheader("Quality score distribution")
                    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(filtered_workspace["quality_score"].dropna(), bins=10, edgecolor="#a8c8e8")
                    ax.set_xlabel("Quality score"); ax.set_ylabel("Observations")
                    fig.tight_layout(); render_chart(fig); plt.close(fig)
            atmospheric = [("Cloud", "cloud_pct"), ("Cloud shadow", "cloud_shadow_pct"), ("High aerosol QA flag", "high_aerosol_pct")]
            available_atmospheric = [(label, field) for label, field in atmospheric if field in filtered_workspace.columns]
            if available_atmospheric:
                st.subheader("Atmospheric and pixel-quality indicators")
                st.dataframe(pd.DataFrame([{"Indicator": label, "Mean flagged pixels (%)": filtered_workspace[field].mean(), "Maximum (%)": filtered_workspace[field].max()} for label, field in available_atmospheric]), use_container_width=True, hide_index=True)
                st.caption("High-aerosol values represent an HLS QA flag class, not a direct measurement of aerosol concentration.")
            st.download_button("Download quality-control report", filtered_workspace.to_csv(index=False).encode("utf-8"), "spectraai_quality_control.csv", "text/csv", key="quality_export")

# ============================================================
# TAB 1 — EXISTING DATASET DASHBOARD
# ============================================================

if active_page == "NASA HLS Dataset":

    st.markdown('<div class="mc-eyebrow">DATA OPERATIONS / NASA HLS S30</div>', unsafe_allow_html=True)
    st.header("Observation Intelligence")

    try:
        data = load_data()

    except Exception as error:
        st.error(f"Unable to load dataset: {error}")
        data = pd.DataFrame()

    if not data.empty:

        st.sidebar.markdown("<div class='mc-section'>Dataset filters</div>", unsafe_allow_html=True)

        areas = sorted(data["area_id"].dropna().unique())

        selected_areas = st.sidebar.multiselect(
            "Select Area IDs",
            areas,
            default=areas,
            key="dataset_areas"
        )

        tiles = sorted(data["tile"].dropna().unique())

        selected_tiles = st.sidebar.multiselect(
            "Select Tiles",
            tiles,
            default=tiles,
            key="dataset_tiles"
        )

        categories = ["Excellent", "Good", "Fair", "Poor"]

        selected_categories = st.sidebar.multiselect(
            "Quality Categories",
            categories,
            default=categories,
            key="dataset_categories"
        )

        filtered = data[
            data["area_id"].isin(selected_areas)
            & data["tile"].isin(selected_tiles)
            & data["quality_category"].isin(selected_categories)
        ].copy()

        if filtered.empty:

            st.warning("No observations match the selected filters.")

        else:

            st.subheader("Dataset Overview")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Observations", len(filtered))

            col2.metric(
                "Average Quality Score",
                f"{filtered['quality_score'].mean():.2f}/100"
            )

            col3.metric(
                "Excellent Observations",
                int(
                    (filtered["quality_category"] == "Excellent").sum()
                )
            )

            col4.metric(
                "Poor Observations",
                int(
                    (filtered["quality_category"] == "Poor").sum()
                )
            )

            st.divider()
            st.header("Quality Assessment")

            left, right = st.columns(2)

            category_order = [
                "Excellent", "Good", "Fair", "Poor"
            ]

            with left:

                st.subheader("Quality Category Distribution")

                counts = (
                    filtered["quality_category"]
                    .value_counts()
                    .reindex(category_order, fill_value=0)
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                counts.plot(kind="bar", ax=ax)

                ax.set_xlabel("Quality Category")
                ax.set_ylabel("Observations")
                ax.tick_params(axis="x", rotation=0)

                fig.tight_layout()
                render_chart(fig)
                plt.close(fig)

            with right:

                st.subheader("Quality Score Distribution")

                fig, ax = plt.subplots(figsize=(7, 4))

                ax.hist(
                    filtered["quality_score"].dropna(),
                    bins=10,
                    edgecolor="black"
                )

                ax.set_xlabel("Quality Score")
                ax.set_ylabel("Observations")

                fig.tight_layout()
                render_chart(fig)
                plt.close(fig)

            st.divider()
            st.header("Observation Quality Ranking")

            ranking = filtered.sort_values(
                "quality_score",
                ascending=False
            ).copy()

            ranking["observation"] = (
                ranking["date_doy"]
                + " | Area "
                + ranking["area_id"]
                + " | "
                + ranking["tile"]
            )

            fig, ax = plt.subplots(figsize=(12, 7))

            ax.barh(
                ranking["observation"],
                ranking["quality_score"]
            )

            ax.axvline(90, linestyle="--", label="Excellent")
            ax.axvline(75, linestyle="--", label="Good")
            ax.axvline(50, linestyle="--", label="Fair")

            ax.set_xlabel("Quality Score")
            ax.set_ylabel("Observation")
            ax.set_xlim(0, 105)
            ax.invert_yaxis()
            ax.legend()

            fig.tight_layout()
            render_chart(fig)
            plt.close(fig)

            st.divider()
            st.header("Multispectral Analysis")

            spectral_columns = [
                "clear_B02_mean",
                "clear_B03_mean",
                "clear_B04_mean",
                "clear_B8A_mean"
            ]

            if all(c in filtered.columns for c in spectral_columns):

                spectral_means = {
                    "Blue (B02)": filtered["clear_B02_mean"].mean(),
                    "Green (B03)": filtered["clear_B03_mean"].mean(),
                    "Red (B04)": filtered["clear_B04_mean"].mean(),
                    "NIR (B8A)": filtered["clear_B8A_mean"].mean()
                }

                fig, ax = plt.subplots(figsize=(9, 4))

                ax.bar(
                    list(spectral_means.keys()),
                    list(spectral_means.values())
                )

                ax.set_ylabel("Mean Surface Reflectance")

                fig.tight_layout()
                render_chart(fig)
                plt.close(fig)

            if "clear_NDVI_mean" in filtered.columns:

                st.subheader("Vegetation Index — NDVI")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Average NDVI",
                    f"{filtered['clear_NDVI_mean'].mean():.3f}"
                )

                col2.metric(
                    "NDVI Standard Deviation",
                    f"{filtered['clear_NDVI_mean'].std():.3f}"
                )

                fig, ax = plt.subplots(figsize=(10, 4))

                ax.hist(
                    filtered["clear_NDVI_mean"].dropna(),
                    bins=10,
                    edgecolor="black"
                )

                ax.set_xlabel("NDVI")
                ax.set_ylabel("Observations")

                fig.tight_layout()
                render_chart(fig)
                plt.close(fig)

            st.divider()
            st.header("Atmospheric Quality Indicators")

            if "high_aerosol_pct" in filtered.columns:

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Average High-Aerosol Flag",
                        f"{filtered['high_aerosol_pct'].mean():.2f}%"
                    )

                    fig, ax = plt.subplots(figsize=(7, 4))

                    ax.hist(
                        filtered["high_aerosol_pct"].dropna(),
                        bins=10,
                        edgecolor="black"
                    )

                    ax.set_xlabel("High Aerosol Flag (%)")
                    ax.set_ylabel("Observations")

                    fig.tight_layout()
                    render_chart(fig)
                    plt.close(fig)

                with col2:

                    fig, ax = plt.subplots(figsize=(7, 4))

                    ax.scatter(
                        filtered["high_aerosol_pct"],
                        filtered["quality_score"]
                    )

                    ax.set_xlabel("High Aerosol Flag (%)")
                    ax.set_ylabel("Quality Score")

                    fig.tight_layout()
                    render_chart(fig)
                    plt.close(fig)

            st.divider()
            st.header("Observation Explorer")

            display_columns = [
                "date_doy",
                "area_id",
                "tile",
                "quality_score",
                "quality_category",
                "clear_pixels_pct",
                "high_aerosol_pct",
                "clear_NDVI_mean"
            ]

            available_columns = [
                column for column in display_columns
                if column in ranking.columns
            ]

            st.dataframe(
                ranking[available_columns].reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Inspect an Observation")

            observation_options = ["All filtered observations"] + ranking["observation"].tolist()
            selected_observation = st.selectbox(
                "Choose an observation to inspect (or keep all)",
                observation_options,
                index=0,
                key="dataset_observation"
            )

            if selected_observation == "All filtered observations":
                # Use numeric means so the same intelligence panel can describe
                # the complete set selected by the sidebar filters.
                selected_row = filtered.mean(numeric_only=True)
                selected_row["quality_category"] = "Dataset average"
                selected_row["observation"] = "All filtered observations"
            else:
                selected_row = ranking[
                    ranking["observation"] == selected_observation
                ].iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Quality Score",
                f"{selected_row['quality_score']:.2f}/100"
            )

            col2.metric(
                "Category",
                selected_row["quality_category"]
            )

            col3.metric(
                "Clear Reflectance Coverage",
                f"{selected_row['clear_reflectance_coverage_pct']:.2f}%"
            )

            st.write(
                "High aerosol flag percentage:",
                f"{selected_row['high_aerosol_pct']:.2f}%"
            )

            st.write(
                "Cloud percentage:",
                f"{selected_row['cloud_pct']:.2f}%"
            )

            st.write(
                "Cloud shadow percentage:",
                f"{selected_row['cloud_shadow_pct']:.2f}%"
            )

            # ------------------------------------------------------------
            # OBSERVATION INTELLIGENCE — plain-language interpretation
            # ------------------------------------------------------------
            st.divider()
            st.header("Observation Intelligence")
            st.caption(
                "A plain-language reading of the selected observation or filtered dataset, "
                "based on its quality and spectral metadata."
            )

            score = float(selected_row.get("quality_score", np.nan))
            clear_coverage = float(
                selected_row.get("clear_reflectance_coverage_pct", np.nan)
            )
            cloud = float(selected_row.get("cloud_pct", np.nan))
            shadow = float(selected_row.get("cloud_shadow_pct", np.nan))
            aerosol = float(selected_row.get("high_aerosol_pct", np.nan))

            if np.isfinite(clear_coverage):
                if clear_coverage >= 90:
                    suitability = "Strong candidate for downstream analysis"
                    suitability_note = (
                        "Most pixels have valid, clear reflectance across the four bands. "
                        "Still check whether the image suits your specific study."
                    )
                elif clear_coverage >= 75:
                    suitability = "Generally usable, with some limitations"
                    suitability_note = (
                        "A useful portion of the image is clear, but inspect masked areas "
                        "before running region-wide analysis."
                    )
                elif clear_coverage >= 50:
                    suitability = "Use selectively"
                    suitability_note = (
                        "A substantial part of the image is obscured or lacks valid "
                        "reflectance. Restrict analysis to suitable clear areas."
                    )
                else:
                    suitability = "Limited suitability"
                    suitability_note = (
                        "Very little of the image has valid clear reflectance. "
                        "Consider another acquisition for broad-area analysis."
                    )
            else:
                suitability = "Not enough quality metadata"
                suitability_note = "The clear-reflectance coverage value is unavailable."

            intel_left, intel_right = st.columns([1.05, 1.4])
            with intel_left:
                st.markdown("**Image suitability**")
                st.metric(
                    "Clear reflectance coverage",
                    f"{clear_coverage:.2f}%" if np.isfinite(clear_coverage) else "N/A"
                )
                st.markdown(f"**{suitability}**")
                st.caption(suitability_note)

            with intel_right:
                st.markdown("**Atmospheric and pixel-quality conditions**")
                condition_rows = []
                for label, value in [
                    ("Cloud-flagged pixels", cloud),
                    ("Cloud-shadow pixels", shadow),
                    ("High-aerosol QA flag", aerosol),
                ]:
                    if np.isfinite(value):
                        condition_rows.append({"Indicator": label, "Pixels": f"{value:.2f}%"})
                if condition_rows:
                    st.dataframe(
                        pd.DataFrame(condition_rows),
                        hide_index=True,
                        use_container_width=True
                    )
                st.caption(
                    "The high-aerosol value is an HLS quality flag class, not a direct "
                    "measurement of atmospheric aerosol concentration."
                )

            st.markdown("**Surface characteristics from spectral indices**")
            index_fields = [
                ("NDVI", "clear_NDVI_mean", "Vegetation contrast indicator"),
                ("NDWI", "clear_NDWI_mean", "Water-related spectral contrast indicator"),
                ("EVI", "clear_EVI_mean", "Enhanced vegetation indicator"),
            ]
            available_indices = [
                (label, float(selected_row[field]), description)
                for label, field, description in index_fields
                if field in selected_row.index and pd.notna(selected_row[field])
            ]
            if available_indices:
                index_cols = st.columns(len(available_indices))
                for index_col, (label, value, description) in zip(index_cols, available_indices):
                    index_col.metric(label, f"{value:.3f}")
                    index_col.caption(description)
            else:
                st.caption("Spectral-index values are not available for this observation.")

            if np.isfinite(score):
                if score >= 75:
                    next_step = (
                        "Proceed with your intended analysis on clear pixels, while "
                        "respecting the cloud and shadow mask."
                    )
                elif score >= 50:
                    next_step = (
                        "Use only the clear portions and compare with another date "
                        "before drawing area-wide conclusions."
                    )
                else:
                    next_step = (
                        "Prefer a clearer acquisition if your task requires broad "
                        "spatial coverage."
                    )
            else:
                next_step = "Review the source quality metadata before using this observation."

            st.info(f"**Suggested next step:** {next_step}")
            st.caption(
                "These indicators support image screening; they do not confirm wildfire, "
                "damage, land-cover change, or other real-world events. Multispectral "
                "anomaly screening is available in Upload & Analyse when the required "
                "satellite bands are supplied."
            )

            st.download_button(
                "Download Filtered Quality Report",
                filtered.to_csv(index=False).encode("utf-8"),
                file_name="spectraai_quality_report.csv",
                mime="text/csv",
                key="dataset_download"
            )

    else:
        st.info(
            "The NASA dataset report is not available. "
            "Check data/metadata/quality_assessment.csv."
        )


# ============================================================
# TAB 2 — UPLOAD AND ANALYSE
# ============================================================

elif active_page in ("Image Analysis", "Anomaly Lab"):

    satellite_mode = "Five separate HLS GeoTIFF layers"

    if active_page == "Image Analysis":
        st.markdown('<div class="mc-eyebrow">IMAGE ANALYSIS / VISUAL INSPECTION</div>', unsafe_allow_html=True)
        st.header("Image Analysis")
        st.caption("Upload an ordinary image for visual quality assessment, feature extraction and an experimental visual outlier scan.")
        analysis_type = "Ordinary Image"
    else:
        st.markdown('<div class="mc-eyebrow">ANOMALY LAB / MULTISPECTRAL SCREENING</div>', unsafe_allow_html=True)
        st.header("Anomaly Lab")
        st.caption(
            "Anomaly screening only: use a single image for visual outlier "
            "screening, or HLS spectral layers for multispectral screening. "
            "Visual quality metrics are available separately in Image Analysis."
        )

        anomaly_input_mode = st.radio(
            "Choose your analysis input",
            [
                "Single image (JPG, PNG, WEBP)",
                "Five separate HLS GeoTIFF layers",
                "One multiband GeoTIFF"
            ],
            horizontal=True,
            key="anomaly_input_mode"
        )

        if anomaly_input_mode == "Single image (JPG, PNG, WEBP)":
            analysis_type = "Ordinary Image"
        else:
            analysis_type = "Satellite Imagery"
            satellite_mode = (
                "Five separate HLS GeoTIFF layers"
                if anomaly_input_mode == "Five separate HLS GeoTIFF layers"
                else "One multiband GeoTIFF"
            )

    # --------------------------------------------------------
    # ORDINARY IMAGE UPLOAD
    # --------------------------------------------------------

    if analysis_type == "Ordinary Image":

        if active_page == "Image Analysis":
            st.subheader("Visual Image Quality Analysis")
        else:
            st.subheader("Single-Image Anomaly Screening")

        uploaded_image = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key=(
                "anomaly_single_image_upload"
                if active_page == "Anomaly Lab"
                else "ordinary_image_upload"
            )
        )

        if uploaded_image is not None:

            try:
                validate_upload_size(
                    uploaded_image, MAX_IMAGE_UPLOAD_MB, "Image"
                )

                with Image.open(
                    BytesIO(uploaded_image.getvalue())
                ) as source_image:
                    width, height = source_image.size
                    if width * height > MAX_IMAGE_PIXELS:
                        raise ValueError(
                            f"Image dimensions are {width:,} × {height:,}. "
                            f"The limit is {MAX_IMAGE_PIXELS:,} pixels."
                        )
                    image = source_image.convert("RGB")


                image_array = np.asarray(image)

                st.image(
                    image,
                    caption=uploaded_image.name,
                    use_container_width=True
                )

                if active_page == "Image Analysis":
                    results = analyse_ordinary_image(image_array)

                    st.divider()
                    st.subheader("Image Quality Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Quality Score",
                        f"{results['quality_score']:.1f}/100"
                    )

                    col2.metric(
                        "Quality Category",
                        results["quality_category"]
                    )

                    col3.metric(
                        "Image Dimensions",
                        f"{image.width} × {image.height}"
                    )

                    st.progress(
                        int(round(results["quality_score"]))
                    )

                    st.caption(
                        "The score is a heuristic visual-quality indicator. "
                        "It is not a trained machine-learning prediction. "
                        "A single ordinary image does not contain the spectral "
                        "bands or Fmask needed for NDVI, NDWI, EVI or HLS quality assessment."
                    )

                    st.divider()
                    st.subheader("Extracted Image Features")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Brightness",
                        f"{results['brightness']:.2f}/255"
                    )

                    col2.metric(
                        "Contrast",
                        f"{results['contrast']:.2f}"
                    )

                    col3.metric(
                        "Sharpness",
                        f"{results['sharpness']:.2f}"
                    )

                    col1, col2 = st.columns(2)

                    col1.metric(
                        "Edge Density",
                        f"{results['edge_density'] * 100:.2f}%"
                    )

                    col2.metric(
                        "High-Frequency Variation",
                        f"{results['noise_indicator']:.2f}"
                    )

                    st.divider()
                    st.subheader("Quality Component Scores")

                    components = pd.DataFrame({
                        "Component": [
                            "Brightness",
                            "Contrast",
                            "Sharpness"
                        ],
                        "Score": [
                            results["brightness_score"],
                            results["contrast_score"],
                            results["sharpness_score"]
                        ]
                    })

                    fig, ax = plt.subplots(figsize=(8, 4))

                    ax.bar(
                        components["Component"],
                        components["Score"]
                    )

                    ax.set_ylim(0, 105)
                    ax.set_ylabel("Component Score")

                    fig.tight_layout()
                    render_chart(fig)
                    plt.close(fig)

                    st.subheader("Analysis and Recommendations")

                    recommendations = []

                    if results["brightness"] < 65:
                        recommendations.append(
                            "The image appears relatively dark. "
                            "Consider improving exposure or illumination."
                        )

                    elif results["brightness"] > 195:
                        recommendations.append(
                            "The image appears relatively bright. "
                            "Check for overexposure or loss of highlights."
                        )

                    if results["contrast"] < 25:
                        recommendations.append(
                            "The image has relatively low contrast."
                        )

                    if results["sharpness"] < 30:
                        recommendations.append(
                            "The image has a low sharpness measurement. "
                            "Inspect for blur or focus problems."
                        )

                    if not recommendations:
                        recommendations.append(
                            "No major issues were detected by the "
                            "basic visual quality checks."
                        )

                    for recommendation in recommendations:
                        st.write("•", recommendation)


                # Experimental visual anomaly scan (separate from quality score).
                st.divider()
                st.subheader("Experimental Visual Anomaly Scan")
                st.caption(
                    "This scan highlights unusual local colour patterns. "
                    "It is not a trained wildfire or disaster detector, and "
                    "highlighted regions require visual verification."
                )

                anomaly_results = detect_visual_outliers(image_array)

                anomaly_col1, anomaly_col2 = st.columns(2)
                anomaly_col1.metric(
                    "Highlighted Area",
                    f"{anomaly_results['anomaly_pct']:.3f}%"
                )
                anomaly_col2.metric(
                    "Candidate Regions",
                    anomaly_results["region_count"]
                )

                st.image(
                    anomaly_results["overlay"],
                    caption="Orange-red overlay = locally unusual image regions",
                    use_container_width=True
                )

                if anomaly_results["region_count"]:
                    st.warning(
                        "Potential visual outliers were highlighted. "
                        "These are not confirmed wildfire or damage detections."
                    )
                    st.dataframe(
                        pd.DataFrame(anomaly_results["regions"]),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info(
                        "No strong local colour outliers were found by this "
                        "experimental scan. This does not prove the scene is "
                        "free of environmental anomalies."
                    )

                mask_image = Image.fromarray(anomaly_results["mask"])
                mask_buffer = BytesIO()
                mask_image.save(mask_buffer, format="PNG")
                st.download_button(
                    "Download Anomaly Mask",
                    mask_buffer.getvalue(),
                    file_name="spectraai_visual_anomaly_mask.png",
                    mime="image/png",
                    key="ordinary_anomaly_mask_download"
                )

                report_data = {
                    "filename": uploaded_image.name,
                    "width": image.width,
                    "height": image.height,
                    "visual_anomaly_area_pct": anomaly_results["anomaly_pct"],
                    "visual_anomaly_candidate_regions": anomaly_results["region_count"],
                }
                if active_page == "Image Analysis":
                    report_data.update(results)
                    report_label = "Download Image Quality & Anomaly Report"
                    report_filename = "spectraai_image_report.csv"
                else:
                    report_label = "Download Visual Anomaly Report"
                    report_filename = "spectraai_visual_anomaly_report.csv"

                report = pd.DataFrame([report_data])
                st.download_button(
                    report_label,
                    report.to_csv(index=False).encode("utf-8"),
                    file_name=report_filename,
                    mime="text/csv",
                    key=(
                        "ordinary_download"
                        if active_page == "Image Analysis"
                        else "anomaly_visual_download"
                    )
                )

            except Exception as error:
                st.error(f"Image analysis failed: {error}")


    # --------------------------------------------------------
    # SATELLITE IMAGE UPLOAD
    # --------------------------------------------------------

    else:

        st.subheader("Multispectral Satellite Quality Analysis")

        st.info(
            "Upload either five matching HLS GeoTIFF layers "
            "(B02, B03, B04, B8A and Fmask), or one compatible "
            "multiband GeoTIFF containing all five layers."
        )

        if satellite_mode == "Five separate HLS GeoTIFF layers":

            uploaded_files = st.file_uploader(
                "Upload B02, B03, B04, B8A and Fmask GeoTIFF files",
                type=["tif", "tiff"],
                accept_multiple_files=True,
                key="hls_layer_upload"
            )

            if uploaded_files:

                total_upload_mb = sum(
                    item.size for item in uploaded_files
                ) / (1024 * 1024)

                if total_upload_mb > MAX_RASTER_TOTAL_MB:
                    st.error(
                        f"The combined upload is {total_upload_mb:.1f} MB. "
                        f"The total limit is {MAX_RASTER_TOTAL_MB} MB."
                    )
                    st.stop()

                layer_files = {}

                for uploaded_file in uploaded_files:

                    try:
                        validate_upload_size(
                            uploaded_file,
                            MAX_RASTER_UPLOAD_MB,
                            uploaded_file.name
                        )
                    except ValueError as upload_error:
                        st.error(str(upload_error))
                        st.stop()

                    layer = detect_hls_layer(uploaded_file.name)

                    if layer is not None:

                        if layer in layer_files:
                            st.error(
                                f"More than one file detected for {layer}."
                            )
                            layer_files = {}
                            break

                        layer_files[layer] = uploaded_file

                required_layers = [
                    "B02", "B03", "B04", "B8A", "Fmask"
                ]

                missing_layers = [
                    layer for layer in required_layers
                    if layer not in layer_files
                ]

                if missing_layers:

                    st.warning(
                        "Missing or unrecognized layers: "
                        + ", ".join(missing_layers)
                        + ". Use the original HLS filenames."
                    )

                else:

                    try:

                        bands = {}
                        band_masks = {}
                        metadata_list = []

                        for layer in required_layers:

                            raw, valid, scale, offset, metadata = (
                                read_uploaded_raster(
                                    layer_files[layer],
                                    band_number=1
                                )
                            )

                            metadata_list.append(metadata)

                            if layer == "Fmask":

                                fmask = raw.astype(np.uint8)
                                fmask_valid = (
                                    valid & (raw != 255)
                                )

                            else:

                                reflectance = (
                                    raw.astype(np.float32) * scale
                                    + offset
                                )

                                reflectance[~valid] = np.nan

                                bands[layer] = reflectance
                                band_masks[layer] = valid

                        validate_raster_grids(metadata_list)

                        results = analyse_satellite_arrays(
                            bands,
                            band_masks,
                            fmask,
                            fmask_valid
                        )

                        st.success("Satellite image analysis completed.")
                        show_spectral_anomaly_results(
                            bands, band_masks, fmask, fmask_valid,
                            key_prefix="hls_upload"
                        )

                        st.subheader("Satellite Quality Results")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Quality Score",
                            f"{results['quality_score']:.2f}/100"
                        )

                        col2.metric(
                            "Quality Category",
                            results["quality_category"]
                        )

                        col3.metric(
                            "Clear Reflectance Coverage",
                            f"{results['clear_reflectance_coverage_pct']:.2f}%"
                        )

                        st.progress(
                            int(round(results["quality_score"]))
                        )

                        st.caption(
                            "Satellite quality score = percentage of total "
                            "pixels with valid reflectance in all four bands "
                            "and no cloud, cloud-adjacency, shadow or snow "
                            "flags in Fmask. Water pixels are retained. "
                            "High aerosol is reported separately."
                        )

                        st.divider()
                        st.subheader("Atmospheric Quality Indicators")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Cloud",
                            f"{results['cloud_pct']:.2f}%"
                        )

                        col2.metric(
                            "Cloud Shadow",
                            f"{results['cloud_shadow_pct']:.2f}%"
                        )

                        col3.metric(
                            "High Aerosol",
                            f"{results['high_aerosol_pct']:.2f}%"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Adjacent to Cloud/Shadow",
                            f"{results['adjacent_cloud_shadow_pct']:.2f}%"
                        )

                        col2.metric(
                            "Snow/Ice",
                            f"{results['snow_ice_pct']:.2f}%"
                        )

                        col3.metric(
                            "Water",
                            f"{results['water_pct']:.2f}%"
                        )

                        st.divider()
                        st.subheader("Spectral Analysis")

                        band_means = results["band_means"]

                        fig, ax = plt.subplots(figsize=(8, 4))

                        ax.bar(
                            ["Blue B02", "Green B03", "Red B04", "NIR B8A"],
                            [
                                band_means["B02"],
                                band_means["B03"],
                                band_means["B04"],
                                band_means["B8A"]
                            ]
                        )

                        ax.set_ylabel("Mean Surface Reflectance")

                        fig.tight_layout()
                        render_chart(fig)
                        plt.close(fig)

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Mean NDVI",
                            f"{results['NDVI_mean']:.4f}"
                        )

                        col2.metric(
                            "Mean NDWI",
                            f"{results['NDWI_mean']:.4f}"
                        )

                        col3.metric(
                            "Mean EVI",
                            f"{results['EVI_mean']:.4f}"
                        )

                        st.caption(
                            "NDWI here uses the Green and NIR bands "
                            "(McFeeters formulation)."
                        )

                        report = pd.DataFrame([{
                            key: value
                            for key, value in results.items()
                            if key != "band_means"
                        }])

                        for band, value in band_means.items():
                            report[f"mean_{band}"] = value

                        st.download_button(
                            "Download Satellite Quality Report",
                            report.to_csv(index=False).encode("utf-8"),
                            file_name="spectraai_satellite_report.csv",
                            mime="text/csv",
                            key="satellite_hls_download"
                        )

                    except Exception as error:
                        st.error(
                            f"Satellite analysis failed: {error}"
                        )


        # ----------------------------------------------------
        # ONE MULTIBAND GEOTIFF
        # ----------------------------------------------------

        else:

            uploaded_raster = st.file_uploader(
                "Upload one multispectral GeoTIFF",
                type=["tif", "tiff"],
                key="multiband_upload"
            )

            if uploaded_raster is not None:

                try:
                    validate_upload_size(
                        uploaded_raster,
                        MAX_RASTER_UPLOAD_MB,
                        "Multiband GeoTIFF"
                    )

                    with MemoryFile(uploaded_raster.getvalue()) as mem:

                        with mem.open() as src:

                            validate_raster_dimensions(
                                src, uploaded_raster.name
                            )
                            band_count = src.count

                    if band_count < 5:

                        st.error(
                            "This raster has fewer than five bands. "
                            "A compatible five-layer dataset is required."
                        )

                    else:

                        st.write(
                            f"Detected raster bands: {band_count}"
                        )

                        st.warning(
                            "Map each raster band to its correct layer. "
                            "The Fmask band must contain the original "
                            "HLS QA bit values, not a rendered colour image."
                        )

                        options = list(range(1, band_count + 1))

                        col1, col2 = st.columns(2)

                        with col1:

                            b02_index = st.selectbox(
                                "Blue — B02 band",
                                options,
                                index=0
                            )

                            b03_index = st.selectbox(
                                "Green — B03 band",
                                options,
                                index=min(1, band_count - 1)
                            )

                            b04_index = st.selectbox(
                                "Red — B04 band",
                                options,
                                index=min(2, band_count - 1)
                            )

                        with col2:

                            b8a_index = st.selectbox(
                                "NIR — B8A band",
                                options,
                                index=min(3, band_count - 1)
                            )

                            fmask_index = st.selectbox(
                                "Fmask QA band",
                                options,
                                index=min(4, band_count - 1)
                            )

                        selected_indices = [
                            b02_index,
                            b03_index,
                            b04_index,
                            b8a_index,
                            fmask_index
                        ]

                        if len(set(selected_indices)) != 5:

                            st.error(
                                "Each layer must use a different raster band."
                            )

                        elif st.button(
                            "Analyse Satellite GeoTIFF",
                            key="analyse_multiband"
                        ):

                            with MemoryFile(
                                uploaded_raster.getvalue()
                            ) as mem:

                                with mem.open() as src:

                                    validate_raster_dimensions(
                                        src, uploaded_raster.name
                                    )

                                    mapping = {
                                        "B02": b02_index,
                                        "B03": b03_index,
                                        "B04": b04_index,
                                        "B8A": b8a_index,
                                        "Fmask": fmask_index
                                    }

                                    bands = {}
                                    band_masks = {}

                                    for layer in [
                                        "B02", "B03", "B04", "B8A"
                                    ]:

                                        index = mapping[layer]

                                        raw = src.read(index)

                                        valid = (
                                            src.read_masks(index) > 0
                                        )

                                        nodata = src.nodatavals[index - 1]

                                        if nodata is not None:
                                            valid &= raw != nodata

                                        scale = src.scales[index - 1]
                                        offset = src.offsets[index - 1]

                                        scale = (
                                            1.0 if scale is None else scale
                                        )

                                        offset = (
                                            0.0 if offset is None else offset
                                        )

                                        reflectance = (
                                            raw.astype(np.float32)
                                            * float(scale)
                                            + float(offset)
                                        )

                                        reflectance[~valid] = np.nan

                                        bands[layer] = reflectance
                                        band_masks[layer] = valid

                                    fmask_raw = src.read(fmask_index)

                                    fmask_valid = (
                                        src.read_masks(fmask_index) > 0
                                    )

                                    fmask_valid &= fmask_raw != 255

                                    fmask = fmask_raw.astype(np.uint8)

                            results = analyse_satellite_arrays(
                                bands,
                                band_masks,
                                fmask,
                                fmask_valid
                            )

                            st.success(
                                "Multiband satellite analysis completed."
                            )
                            show_spectral_anomaly_results(
                                bands, band_masks, fmask, fmask_valid,
                                key_prefix="multiband_upload"
                            )

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Quality Score",
                                f"{results['quality_score']:.2f}/100"
                            )

                            col2.metric(
                                "Category",
                                results["quality_category"]
                            )

                            col3.metric(
                                "Clear Coverage",
                                f"{results['clear_reflectance_coverage_pct']:.2f}%"
                            )

                            st.progress(
                                int(round(results["quality_score"]))
                            )

                            st.subheader("Atmospheric Quality Indicators")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "Cloud",
                                f"{results['cloud_pct']:.2f}%"
                            )

                            col2.metric(
                                "Cloud Shadow",
                                f"{results['cloud_shadow_pct']:.2f}%"
                            )

                            col3.metric(
                                "High Aerosol",
                                f"{results['high_aerosol_pct']:.2f}%"
                            )

                            st.subheader("Spectral Indices")

                            col1, col2, col3 = st.columns(3)

                            col1.metric(
                                "NDVI",
                                f"{results['NDVI_mean']:.4f}"
                            )

                            col2.metric(
                                "NDWI",
                                f"{results['NDWI_mean']:.4f}"
                            )

                            col3.metric(
                                "EVI",
                                f"{results['EVI_mean']:.4f}"
                            )

                            report = pd.DataFrame([{
                                key: value
                                for key, value in results.items()
                                if key != "band_means"
                            }])

                            for band, value in results["band_means"].items():
                                report[f"mean_{band}"] = value

                            st.download_button(
                                "Download Satellite Quality Report",
                                report.to_csv(index=False).encode("utf-8"),
                                file_name="spectraai_satellite_report.csv",
                                mime="text/csv",
                                key="satellite_multiband_download"
                            )

                except Exception as error:
                    st.error(f"Multiband satellite analysis failed: {error}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SpectraAI | Research Prototype | NASA HLS S30 v2.0 | "
    "Ordinary-image scores are heuristic indicators. "
    "Satellite scores represent clear reflectance coverage "
    "based on the uploaded HLS Fmask layer. "
    "Neither score is a validated machine-learning prediction."
)
