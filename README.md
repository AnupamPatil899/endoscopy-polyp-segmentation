# Clinical Polyp Segmentation & Cross-Center Generalization Study

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Google Cloud Run](https://img.shields.io/badge/GCP-Cloud_Run-4285F4.svg)](https://cloud.google.com/run)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clinical deep learning system for real-time colonoscopy polyp segmentation, probability calibration, and multi-tier distribution shift assessment across independent hospital centers.

---

## 📌 Key Highlights & Results

- **In-Distribution Performance (Kvasir-SEG Val, N=200):**
  - **Mean Dice:** `0.9050` (90.50%) | **Median Dice:** `0.9569` (95.69%) | **Mean IoU:** `0.8480`
  - **Polyp Detection Sensitivity:** `87.93%` | **Small Polyp Recall (<5% area):** `93.02%`
- **Zero-Shot Out-of-Distribution Generalization (CVC-ClinicDB, N=612):**
  - **Mean Dice:** `0.8559` (85.59%) | **Median Dice:** `0.9263` (92.63%) | **Generalization Drop:** `-5.43%`
  - **Polyp Detection Sensitivity:** `88.43%` | **Small Polyp Recall (<5% area):** `90.00%`
  - **Expected Calibration Error (ECE):** `0.0179` | **Brier Score:** `0.0210`
- **The "Tail Drop" Finding:**
  - While median Dice dropped by only **3.20%** under domain shift, the **10th percentile (P10 worst cases) dropped by 13.62%** ($0.7688 \rightarrow 0.6641$), proving why medical AI requires percentile tail-risk reporting rather than aggregate means alone.
- **Production-Ready Artifacts:**
  - Interactive **Streamlit Web Application** with real-time thresholding, entity quantification, probability heatmaps, and error maps.
  - Automated **CI/CD Pipeline** deploying to **Google Cloud Run** via GitHub Actions.

---

## 🔬 Multi-Tier Benchmark Comparison

```
+---------------------------+-----------------------+-----------------------+-------------------+
| Metric                    | In-Distribution       | Out-of-Distribution   | Generalization    |
|                           | (Kvasir-SEG Val, N=200)| (CVC-ClinicDB, N=612) | Delta / Drop      |
+---------------------------+-----------------------+-----------------------+-------------------+
| Mean Dice Score           | 0.9050 (90.50%)       | 0.8559 (85.59%)       | -0.0491 (-5.43%)  |
| Median Dice Score         | 0.9569 (95.69%)       | 0.9263 (92.63%)       | -0.0306 (-3.20%)  |
| Worst-Decile (P10) Dice   | 0.7688 (76.88%)       | 0.6641 (66.41%)       | -0.1047 (-13.62%) |
| Mean IoU (Jaccard)        | 0.8480 (84.80%)       | 0.7816 (78.16%)       | -0.0664 (-7.84%)  |
| Median IoU                | 0.9174 (91.74%)       | 0.8626 (86.26%)       | -0.0548 (-5.97%)  |
| Mean Precision            | 0.9240                | 0.8885                | -0.0355 (-3.84%)  |
| Mean Recall               | 0.9167                | 0.8738                | -0.0429 (-4.68%)  |
| Polyp Detection Recall    | 87.93% (204/232)      | 88.43% (627/709)      | +0.50%            |
| Small Polyp Recall (<5%)  | 93.02% (40/43)        | 90.00% (234/260)      | -3.02%            |
| Expected Calib. Error(ECE)| 0.0157                | 0.0179                | +0.0022           |
| Brier Score               | 0.0209                | 0.0210                | +0.0001           |
+---------------------------+-----------------------+-----------------------+-------------------+
```

---

## 🏗️ Architecture & Methodology

```
                   ┌────────────────────────────────────────┐
                   │    Input Colonoscopy Frame (352x352)   │
                   └──────────────────┬─────────────────────┘
                                      │
                         ┌────────────▼───────────┐
                         │  ResNet34 Pretrained   │ (Encoder)
                         │   Feature Extractor    │
                         └────────────┬───────────┘
                                      │ (Skip Connections)
                         ┌────────────▼───────────┐
                         │     U-Net Decoder      │ (Feature Reconstruction)
                         │   + Upsampling Blocks  │
                         └────────────┬───────────┘
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            │                                                   │
  ┌─────────▼─────────┐                               ┌─────────▼─────────┐
  │ Raw Logits Output │                               │  Sigmoid Probs    │
  └─────────┬─────────┘                               └─────────┬─────────┘
            │                                                   │
┌───────────▼───────────┐                           ┌───────────▼───────────┐
│ Combined Dice+BCE Loss│                           │ Post-Processing &     │
│ (Training Signal)     │                           │ Connected Components  │
└───────────────────────┘                           └───────────────────────┘
```

1. **Model:** U-Net with ImageNet-pretrained ResNet34 encoder.
2. **Loss:** Hybrid $\mathcal{L}_{total} = 1.0 \cdot \mathcal{L}_{Dice} + 1.0 \cdot \mathcal{L}_{BCE}$ for scale-invariant overlap and smooth gradient optimization.
3. **Augmentation:** Domain-specific optical pipeline (specular highlight simulation, motion blur, brightness/contrast jitter, affine transforms).
4. **Post-Processing:** Connected-component labeling (`scipy.ndimage`) for discrete lesion counting and bounding box localization.

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
# Clone repository
git clone https://github.com/<username>/polyp-segmentation.git
cd polyp-segmentation

# Setup virtual environment with uv or standard venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Interactive Web App (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

### 3. Command-Line Inference
```bash
# Run inference on a sample image
python src/infer.py \
  --image Data/CVC-ClinicDB/images/100.png \
  --mask Data/CVC-ClinicDB/masks/100.png \
  --threshold 0.50 \
  --output_dir outputs/inference_results
```

### 4. Run Model & Loss Unit Tests
```bash
python src/test_model_and_loss.py
```

---

## 🐳 Docker & Cloud Run Deployment

### Local Container Run:
```bash
docker build -t polyp-segmentation-app:latest .
docker run -p 8080:8080 polyp-segmentation-app:latest
```

### Automated CI/CD (GitHub Actions to Google Cloud Run):
The repository includes `.github/workflows/deploy.yml` configured to:
1. Run automated syntax checks and unit tests.
2. Authenticate to GCP using Workload Identity Federation.
3. Build and push image to Google Artifact Registry.
4. Deploy the Streamlit app to Google Cloud Run (`2 CPU`, `2GB RAM`, port `8080`).

---

## 📂 Repository Structure

```
polyp-segmentation/
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD Cloud Run deployment workflow
├── app/
│   └── streamlit_app.py            # Interactive clinical web application
├── src/
│   ├── model.py                    # PolypUNet architecture definition
│   ├── losses.py                   # Dice, BCE, Combined loss & batch metrics
│   ├── dataset.py                  # PyTorch Dataset & Albumentations pipeline
│   ├── train.py                    # Training engine with AdamW & Cosine Annealing
│   ├── evaluate.py                 # Multi-tier evaluation engine
│   ├── infer.py                    # Modular PolypPredictor & CLI tool
│   └── test_model_and_loss.py      # Unit test suite
├── notebooks/
│   └── failure_analysis.ipynb      # 6-section failure analysis & narrative
├── docs/
│   ├── ADR.md                      # Architectural Decision Records (ADR-001 - ADR-010)
│   ├── EVALUATION.md               # In-distribution vs OOD clinical report
│   ├── FAILURE_ANALYSIS.md         # Clinical failure taxonomy & error analysis
│   ├── DEPLOYMENT.md               # Deployment and inference guide
│   └── DATASETS.md                 # Dataset specifications & split manifests
├── outputs/
│   ├── checkpoints/                # Best trained model weights (.pth)
│   ├── eval_in_distribution/       # Kvasir-SEG validation metrics & plots
│   └── eval_ood_cvc_clinicdb/      # CVC-ClinicDB zero-shot metrics & plots
├── Dockerfile                      # Cloud Run container definition
├── docker-compose.yml              # Local container execution
├── requirements.txt                # Pinned dependency requirements
└── README.md                       # Main repository overview
```

---

## 📖 In-Depth Study Documentation

For deep technical dives into theory, mathematical formulations, and interview guides, explore:
- [`docs/ADR.md`](docs/ADR.md) — Architecture Decision Records
- [`docs/EVALUATION.md`](docs/EVALUATION.md) — Clinical Evaluation Report
- [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md) — Failure Taxonomy
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Deployment Guide

---

## 📜 License
This project is licensed under the MIT License.
