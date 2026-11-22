# SpectraAI 🛰️
### AI-Powered Remote Sensing & Satellite Image Analysis Platform

**SpectraAI** is an interactive remote-sensing platform built with Python and Streamlit to explore satellite imagery, analyse spectral information, assess image quality, and identify potential anomalies.

The project uses NASA Harmonized Landsat Sentinel-2 (HLS) data and combines geospatial processing, spectral indices, image-quality assessment, and machine-learning experimentation in one dashboard.

🌐 **Live Application:** [Open SpectraAI](https://spectraai-kerke9yflj6ikvrvwoh5dy.streamlit.app/)

📂 **GitHub Repository:** https://github.com/krutaswi/SpectraAI

---

## ✨ Key Features

SpectraAI brings multiple remote-sensing workflows together in a single interactive application.

### 1. Observations
- Explore prepared satellite observation metadata.
- Review available observation records and acquisition information.
- Examine the study dataset through interactive visualisations.

### 2. Spectral Lab
- Explore spectral information from satellite imagery.
- Analyse reflectance values across spectral bands.
- Calculate and visualise vegetation and water-related indices.

### 3. Quality Control
- Assess satellite image quality.
- Examine image-quality indicators and quality metrics.
- Explore quality-related patterns through visualisations.

### 4. Anomaly Lab
- Upload individual images for visual anomaly screening.
- Upload five separate HLS GeoTIFF layers.
- Upload a multiband GeoTIFF for supported analysis.
- Explore potential irregularities in satellite imagery.

### 5. Image Analysis
- Analyse uploaded images.
- Examine brightness, contrast, sharpness, and other visual-quality indicators.
- Visualise image characteristics through interactive outputs.

---

## 🛰️ Dataset

The project uses **NASA Harmonized Landsat Sentinel-2 (HLS) S30** satellite observations.

| Dataset detail | Information |
|---|---|
| Data source | NASA HLS S30 |
| Study locations | Ahmedabad, Surat, and Gir, Gujarat, India |
| Observation period | January 1–31, 2025 |
| Complete observations | 34 |
| Spectral and mask layers | 170 GeoTIFF files |
| Layers per observation | 5 |

The five layers used in the prepared observations are:

- B02 – Blue
- B03 – Green
- B04 – Red
- B8A – Narrow Near Infrared
- Fmask – Quality and cloud mask information

The repository contains prepared metadata and preview assets rather than the complete raw satellite raster collection.

---

## 🔬 Spectral Analysis

Spectral analysis uses satellite reflectance information to derive useful indicators of surface characteristics.

Indices explored in the project include:

| Index | Purpose |
|---|---|
| NDVI | Vegetation monitoring |
| NDWI | Water-related feature analysis |
| EVI | Enhanced vegetation assessment |

The platform also explores image characteristics such as brightness, contrast, sharpness, edge density, and noise indicators.

---

## 🤖 Machine Learning & Quality Assessment

SpectraAI includes a machine-learning experimentation workflow for satellite image-quality assessment.

The project explores:
- Spectral and statistical feature extraction.
- Image-quality indicators.
- Synthetic image degradation.
- Random Forest classification.
- Acquisition-level grouped training and testing.

### Benchmark Results

The experimental benchmark compared an original model with an HLS-oriented candidate model.

| Metric | Original model | HLS candidate |
|---|---:|---:|
| Accuracy | 60.48% | 78.63% |
| Macro F1-score | 0.38 | 0.72 |

**Evaluation setup:**
- 15 training acquisition groups.
- 5 testing acquisition groups.
- 26 training tile observations.
- 8 testing tile observations.
- 806 training samples.
- 248 testing samples.
- No shared acquisition groups between training and testing.

**Important limitation:** These results come from a synthetic degradation benchmark. They should not be interpreted as independently verified real-world satellite-quality performance. The held-out evaluation contains only five acquisition groups, so broader validation is needed.

The experimental candidate model is kept separate from the original model. Benchmark results do not automatically imply that the candidate model powers every live dashboard feature.

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Dashboard | Streamlit |
| Data processing | NumPy, Pandas |
| Geospatial processing | Rasterio |
| Machine learning | Scikit-learn |
| Image processing | OpenCV, Pillow |
| Visualisation | Matplotlib |
| Model persistence | Joblib |
| Version control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```text
SpectraAI/
│
├── spectra_dashboard.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── metadata/
│   └── previews/
│
├── model/
│
└── .streamlit/
    └── config.toml
```

The repository also contains supporting scripts for data preparation, validation, feature extraction, and model experimentation.

---

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/krutaswi/SpectraAI.git
cd SpectraAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the dashboard

```bash
streamlit run spectra_dashboard.py
```

The application will open in your browser.

---

## ☁️ Deployment

SpectraAI is deployed using **Streamlit Community Cloud** and connected to this GitHub repository.

The deployed application can be updated by pushing new commits to the connected GitHub branch.

---

## 🔮 Future Improvements

Potential directions for further development include:

- Expanded real-world validation datasets.
- Additional satellite spectral bands and indices.
- More extensive machine-learning evaluation.
- Improved anomaly detection methods.
- Support for additional satellite data sources.
- More advanced geospatial visualisation and comparison tools.

---

## 👩‍💻 Author

**Krutaswi Bhalani**

Computer Engineering | AI/ML | Remote Sensing | Software Development

---

*SpectraAI — Turning satellite observations into meaningful insights.* 🛰️
