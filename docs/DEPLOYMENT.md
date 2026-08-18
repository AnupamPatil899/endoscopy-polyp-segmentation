# Deployment & Inference Guide

This guide details running inference locally via CLI or Web Application, containerizing with Docker, and automated CI/CD deployment to Google Cloud Run.

---

## 1. Interactive Web Application (Streamlit)

```bash
# Launch Streamlit application locally
.venv/bin/streamlit run app/streamlit_app.py --server.port 8501
```

### Features:
- Preset selector (in-distribution & OOD demo cases) + custom file upload.
- Dynamic threshold slider ($\tau \in [0.05, 0.95]$) and opacity adjustment.
- 4-panel visualizer (Raw frame, Segmentation overlay with bounding boxes, Probability heatmap, and TP/FP/FN error map).
- Clinical telemetry dashboard (polyp count, area %, size bucket, latency, live Dice/IoU).

---

## 2. Command-Line Inference (`src/infer.py`)

### Single Image Inference:
```bash
python src/infer.py \
  --image Data/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg \
  --output_dir outputs/inference_demo
```

### Inference with Ground Truth Mask & Error Diagnostics:
```bash
python src/infer.py \
  --image Data/CVC-ClinicDB/images/100.png \
  --mask Data/CVC-ClinicDB/masks/100.png \
  --threshold 0.50 \
  --output_dir outputs/inference_demo
```

---

## 3. Docker Containerization

```bash
# Build Docker image
docker build -t polyp-segmentation-app:latest .

# Run container locally
docker run -p 8080:8080 polyp-segmentation-app:latest
```

---

## 4. Automated CI/CD to Google Cloud Run

The repository includes a GitHub Actions workflow in [`.github/workflows/deploy.yml`](file:///home/anupa/polyp-segmentation/.github/workflows/deploy.yml):

1. **CI Stage**: Runs syntax compilation checks (`py_compile`) and unit tests (`src/test_model_and_loss.py`).
2. **CD Stage**: Authenticates to Google Cloud via Workload Identity Federation, builds the image with Buildx layer caching, pushes to Google Artifact Registry, and deploys to Cloud Run with 2 CPU, 2GB RAM on port 8080.

### Required GitHub Repository Secrets:
- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_SERVICE_ACCOUNT`
