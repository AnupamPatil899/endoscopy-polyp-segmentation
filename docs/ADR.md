# Architecture Decision Records (ADR)

This log records the engineering and architectural decisions for the Polyp Segmentation & Distribution Shift project.

---

## ADR-001: Environment Management & Tooling (uv + Virtualenv)
- **Status:** Accepted / Implemented
- **Context:** Need a fast, deterministic, reproducible Python environment for PyTorch, Albumentations, SMP, and Streamlit.
- **Decision:** Use `uv` with Python 3.12 and explicit pinned minimum requirements in `requirements.txt`.
- **Rationale:** `uv` provides sub-second virtualenv creation and package resolution, preventing dependency conflicts and environment drift.
- **Trade-offs:** Requires `uv` binary; standard `pip` remains a viable fallback.
- **Impact:** Consistent execution across training scripts, evaluation pipelines, and container builds.

---

## ADR-002: Model Architecture Selection (U-Net with Pretrained ResNet34)
- **Status:** Accepted / Implemented
- **Context:** Must establish a robust segmentation baseline without overcomplicating the model search space.
- **Decision:** Standard U-Net with ImageNet-pretrained ResNet34 encoder via `segmentation_models_pytorch`.
- **Rationale:**
  1. *Clinical Standard:* U-Net's encoder-decoder structure with skip connections preserves fine spatial boundaries lost during downsampling, critical for identifying precise polyp margins.
  2. *Attribution Isolation:* Using an established, stable architecture ensures that generalization drops on OOD datasets (CVC-ClinicDB) are attributable to domain/covariate shift rather than architectural instability.
  3. *Pretrained Feature Transfer:* Early convolutional filters (edges, textures, gradients) transfer effectively to endoscopic imagery, allowing faster convergence on the limited ~1,000 image dataset.
- **Trade-offs:** ResNet34 has a fixed local receptive field compared to modern vision transformers (SegFormer); transformer heads are reserved for future work.
- **Impact:** Low training overhead, high reproducibility, easy baseline comparison.

---

## ADR-003: Loss Function Formulation (Combined Dice + Binary Cross-Entropy)
- **Status:** Accepted / Implemented
- **Context:** Endoscopic images suffer from heavy foreground-background class imbalance (polyps often occupy <5-10% of the frame).
- **Decision:** Hybrid loss: $\mathcal{L}_{total} = \alpha \mathcal{L}_{Dice} + \beta \mathcal{L}_{BCE}$ (default $\alpha=1.0, \beta=1.0$).
- **Rationale:**
  1. *Plain BCE failure mode:* Dominated by easy true negatives (background mucosa/lumen), leading to deceptively low loss while failing to segment small polyps.
  2. *Plain Dice failure mode:* Gradients can be unstable or near zero in early training when predicted foreground overlap is minimal.
  3. *Combined benefit:* BCE provides smooth, stable gradients from step 0; Dice provides scale-invariant overlap optimization directly aligned with the target evaluation metric.
- **Trade-offs:** Requires balanced weighting ($\alpha, \beta$); loss values are composite rather than purely probabilistic.
- **Impact:** Faster convergence, robust boundary learning, and prevention of trivial background-only predictions.

---

## ADR-004: Domain-Specific Optical Augmentation Strategy
- **Status:** Accepted / Implemented
- **Context:** Models trained on single-center colonoscopy data overfit to specific camera sensors, illumination geometries, and mucosal texture hues.
- **Decision:** Include standard affine transforms alongside domain-specific optical augmentations:
  - **Motion Blur:** Simulates rapid colonoscope tip movement and peristaltic motion.
  - **Brightness & Contrast Jitter / Color Perturbation:** Simulates varying light source intensities and mucosal color tones.
  - **Gaussian Noise & Texture Distortions:** Regularizes against sensor noise.
- **Rationale:** Directly addresses physical failure modes of colonoscopy cameras, regularizing the encoder against scope-specific artifacts.
- **Trade-offs:** Excessive augmentation can distort realistic polyp morphology; parameters must be calibrated.
- **Impact:** Improved out-of-distribution robustness when transferring to unseen hospital datasets.

---

## ADR-005: Strict OOD Isolation Protocol (CVC-ClinicDB)
- **Status:** Accepted / Implemented
- **Context:** Project goal is measuring real-world cross-center generalization degradation.
- **Decision:** CVC-ClinicDB is strictly treated as a locked, held-out test set. Zero hyperparameter tuning, zero validation checks, and zero early stopping decisions touch CVC-ClinicDB.
- **Rationale:** Any interaction with CVC-ClinicDB during training or tuning compromises the scientific integrity of the generalization gap study.
- **Trade-offs:** All model selection decisions are validated solely on Kvasir-SEG validation splits.
- **Impact:** Valid, defensible distribution shift findings suitable for clinical scrutiny.

---

## ADR-006: Multi-Tiered Evaluation Framework
- **Status:** Accepted / Implemented
- **Context:** Standard mean Dice metrics hide critical tail-risk clinical failures.
- **Decision:** Implement three distinct evaluation tiers in a unified evaluation script:
  1. *Tier 1 (Pixel-level):* Mean, Median, and 10th-percentile (worst-decile) Dice and IoU, Precision, Recall.
  2. *Tier 2 (Polyp-level Detection):* Connected-component detection sensitivity (did the model find each individual polyp independent of pixel boundary overlap?).
  3. *Tier 3 (Calibration & Uncertainty):* Expected Calibration Error (ECE), Brier score, and reliability diagram binning on raw probabilities.
  4. *Stratification:* Granular breakdown across Small (<5%), Medium (5-20%), and Large (>20%) polyps.
- **Rationale:** Clinicians care primarily about polyp detection (did we miss the lesion?); safety systems depend on whether predicted confidence correlates with true accuracy.
- **Trade-offs:** More computational overhead during evaluation passes; requires connected-component post-processing.
- **Impact:** Comprehensive diagnostic narrative that goes beyond simple academic benchmark tables.

---

## ADR-007: Targeted Hyperparameter Ablation Strategy
- **Status:** Accepted / Implemented
- **Context:** Large brute-force grid searches in computer vision are computationally wasteful when standard baselines exist.
- **Decision:** Use hypothesis-driven parameter selection (differential learning rates, cosine annealing schedule with warmup, weight decay).
- **Rationale:** Pretrained ResNet34 starts in a well-behaved loss basin; resources are better allocated to thorough cross-center evaluation and failure analysis.
- **Trade-offs:** Tests fewer combinations but yields cleaner, interpretable insights.
- **Impact:** Efficient training (8.87 minutes on GPU), reproducible results.

---

## ADR-008: Clinical Failure Taxonomy & Tail-Risk Diagnostics
- **Status:** Accepted / Implemented
- **Context:** Evaluation revealed that mean Dice metrics mask severe tail degradation on specific morphological and optical edge cases.
- **Decision:** Establish a 5-category clinical failure taxonomy and embed high-resolution 5-panel diagnostic visualizations (RGB, GT, Pred, Heatmap, Error Map) in `notebooks/failure_analysis.ipynb`:
  1. *Specular Glare & Reflection:* Intensity saturation causing boundary truncation.
  2. *Flat / Sessile Polyp Morphology (Paris IIa/IIb):* Low-elevation lesions lacking shadow margins.
  3. *Motion & Focus Blur:* Scope translation destroying high-frequency edge gradients.
  4. *Low-Contrast Boundary Ambiguity:* Lesion hues blending into folded mucosal walls.
  5. *Sub-1% Micro-Polyps:* Diminutive lesions suffering feature vanishing in multi-scale encoders.
- **Rationale:** Provides structured, qualitative explanations for quantitative generalization drops, enabling targeted engineering countermeasures.
- **Trade-offs:** Requires sample inspection and domain mapping alongside automated metric calculation.
- **Impact:** Direct translation from ML benchmarks to clinical safety requirements.

---

## ADR-009: Real-Time Inference Engine & Clinical Web Interface Architecture
- **Status:** Accepted / Implemented
- **Context:** Endoscopy systems require fast inference, interactive threshold modulation ($\tau$), lesion quantification (area %, count), and visual error analysis.
- **Decision:** Decouple inference logic into a reusable `PolypPredictor` engine (`src/infer.py`) and an interactive Streamlit dashboard (`app/streamlit_app.py`).
- **Rationale:** Separating the predictor logic from the UI framework allows seamless reuse in CLI batch scripts, video stream workers, or REST API endpoints while delivering an interactive clinician-facing application.
- **Trade-offs:** Running Streamlit introduces browser process overhead; CPU inference takes ~280ms vs ~12ms on GPU.
- **Impact:** Complete end-to-end user-facing application ready for clinical demonstrations and review.

---

## ADR-010: Containerized Deployment & Automated CI/CD (Google Cloud Run + GitHub Actions)
- **Status:** Accepted / Implemented
- **Context:** Need serverless, reproducible cloud hosting for the interactive clinical dashboard with automated GitHub testing and deployment.
- **Decision:** Containerize using Debian-slim (`python:3.11-slim`) with CPU-optimized PyTorch wheels, and automate deployment via GitHub Actions (`.github/workflows/deploy.yml`) to Google Cloud Run:
  - *CI Stage:* Runs automated syntax validation (`py_compile`) and unit tests (`src/test_model_and_loss.py`).
  - *CD Stage:* Authenticates via GCP Workload Identity Provider / Service Account, builds image with Buildx layer caching, pushes to Google Artifact Registry, and deploys to Cloud Run on Port 8080.
- **Rationale:** Serverless container scaling provides zero idle cost, automatic HTTPS endpoints, isolated execution, and fast container cold starts (~3-5s).
- **Trade-offs:** CPU inference latency is ~280ms per frame (suitable for interactive web examination, while real-time 60fps video requires GPU instances).
- **Impact:** Production-grade deployment pipeline matching enterprise cloud standards.
