
# SpectraAI
### Satellite Image Quality Intelligence & Multispectral Anomaly Screening

SpectraAI is an AI/ML-powered remote-sensing application built using NASA Harmonized Landsat and Sentinel-2 (HLS) data. It combines multispectral analysis, satellite image quality assessment, visual feature extraction, and anomaly screening in an interactive Streamlit dashboard.

The project is designed to make satellite imagery analysis more accessible through an interactive research dashboard and user-uploaded image analysis.

---

## Project Overview

SpectraAI explores satellite observations and evaluates image quality using spectral information, Fmask-based quality control, image-processing techniques, and experimental machine-learning models.

The initial research dataset covers selected regions of Gujarat, India:

- Ahmedabad
- Surat
- Gir

**Study period:** January 1–31, 2025

**Dataset:** NASA HLS S30

**Dataset size:** 34 complete tile observations, represented by 170 TIFF layers.

Each observation includes:

- B02 – Blue
- B03 – Green
- B04 – Red
- B8A – Near Infrared
- Fmask – Cloud, shadow, snow, water and quality information

---

## Dashboard Workspaces

SpectraAI includes five interactive workspaces.

### 1. Observations
Explore the prepared HLS observation dataset, acquisition information, and observation-level summaries.

### 2. Spectral Lab
Explore multispectral information and spectral indices, including:

- NDVI – Normalized Difference Vegetation Index
- NDWI – Normalized Difference Water Index
- EVI – Enhanced Vegetation Index
- Band reflectance statistics

### 3. Quality Control
Evaluate satellite observation quality using Fmask classifications and clear-reflectance coverage.

### 4. Anomaly Lab
Perform anomaly screening using supported image and satellite data inputs.

Supported input modes:

- Single ordinary image (JPG, PNG, WEBP)
- Five separate HLS GeoTIFF layers
- One multiband GeoTIFF

Ordinary RGB images support visual anomaly screening but do not provide the multispectral information required for HLS-derived spectral indices or Fmask-based quality assessment.

### 5. Image Analysis
Evaluate ordinary images using visual features such as:

- Brightness
- Contrast
- Sharpness
- Edge density
- Noise indicator

---

## Methodology

### Fmask-based Quality Assessment

Fmask information is used to identify cloud, adjacent-to-cloud, shadow, snow, water, aerosol and no-data conditions.

The project's clear-reflectance coverage assessment excludes cloud, adjacent, shadow, snow and no-data pixels. Water is retained as a valid surface class, while aerosol information is evaluated separately.

### Multispectral Analysis

Spectral indices are calculated using the required reflectance bands.

Examples include:

- NDVI: vegetation-related analysis using red and near-infrared reflectance.
- NDWI: water-related analysis using green and near-infrared reflectance.
- EVI: vegetation analysis using blue, red and near-infrared reflectance.

### Visual Feature Extraction

Image-processing techniques extract brightness, contrast, sharpness, edge density and noise-related features.

### Machine Learning

Random Forest classification was explored for image-quality categorization.

The project distinguishes between rule-based satellite quality assessment and experimental machine-learning classification. They represent different approaches and should not be interpreted as interchangeable.

---

## Dataset Validation

The prepared dataset validation produced the following results:

| Validation | Result |
|---|---:|
| Complete observations | 34 |
| Fmask validation matches | 34 |
| Fmask mismatches | 0 |
| Incomplete observations | 0 |

These results describe the prepared research dataset.

---

## Machine Learning Evaluation

An experimental HLS-trained candidate model was compared with the original baseline model using the same held-out acquisition groups.

| Metric | Original Model | HLS Candidate |
|---|---:|---:|
| Accuracy | 60.48% | 78.63% |
| Macro F1-score | 0.38 | 0.72 |

The evaluation used an acquisition-level grouped split to prevent tiles from the same acquisition from appearing in both training and testing sets.

The held-out evaluation contained 248 samples from five acquisition groups.

**Important limitation:** The benchmark uses synthetic image degradations. These results are an internal experimental evaluation, not independent validation against manually labelled real-world image-quality data. The relatively small number of held-out acquisition groups also limits generalization claims.

---

## Technology Stack

- Python
- Streamlit
- NumPy
- Pandas
- Rasterio
- OpenCV
- Scikit-learn
- Matplotlib
- Pillow
- Joblib

---

## Project Structure

```text
SpectraAI/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── metadata/
│   ├── previews/
│   └── raw/
│       └── HLS_S30/
│
├── model/
│
├── spectra_dashboard.py
├── image_features.py
├── app.py
│
├── analyze_fmask.py
├── dataset_inventory.py
├── extract_multispectral_features.py
├── inspect_fmask_distribution.py
├── quality_assessment.py
│
├── train_model.py
├── train_compare_quality_model.py
├── validate_fmask_nasa.py
├── validate_observations.py
├── validate_quality_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

Raw satellite imagery and local virtual environments should not be committed to GitHub.

---

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SpectraAI
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run spectra_dashboard.py
```

---

## Deployment

SpectraAI is designed for deployment using Streamlit Community Cloud.

Deployment configuration:

- Main application: `spectra_dashboard.py`
- Python version: 3.12
- Dependencies: `requirements.txt`
- Streamlit configuration: `.streamlit/config.toml`

The application provides a prepared demonstration experience alongside upload-based analysis.

Hosted environments have finite memory and processing resources. Upload and raster-dimension limits are therefore enforced.

---

## Limitations

- The initial research dataset covers a limited geographic area and time period.
- The experimental ML benchmark uses synthetic degradations rather than independent real-world labels.
- Ordinary RGB images cannot provide HLS-specific spectral indices or Fmask quality information.
- Large satellite rasters may exceed hosted application memory limits.
- Anomaly screening should be treated as an analytical aid rather than a definitive scientific conclusion.

---

## Future Scope

- Expand the dataset across additional geographic regions and seasons.
- Evaluate models using independently labelled real-world imagery.
- Explore temporal anomaly detection.
- Support additional multispectral bands and satellite products.
- Improve scalability for larger geospatial datasets.

---

## Data Acknowledgement

This project uses NASA Harmonized Landsat and Sentinel-2 (HLS) data products.

Refer to the official NASA HLS documentation and applicable data-use terms when obtaining and redistributing source imagery.

---

**SpectraAI — Turning satellite observations into meaningful insights.**
