# Dataset Inventory & Specifications

## 1. Datasets Overview

| Dataset | Role | Size | Source | Characteristics |
|---|---|---|---|---|
| **Kvasir-SEG** | In-Distribution (Train / Val) | 1,000 images & masks | Vestre Viken Hospital Trust, Norway | Multi-resolution white-light colonoscopy |
| **CVC-ClinicDB** | Out-of-Distribution (Held-Out Test) | 612 frames & masks | Hospital Clinic Barcelona, Spain | Video sequence frames, uniform 384x288 |

---

## 2. In-Distribution: Kvasir-SEG
- **Resolution:** Widths: [332 to 1,920 px], Heights: [352 to 1,072 px].
- **Mask Binarization:** Strictly thresholded `(mask > 127).astype(np.float32)` to eliminate boundary JPEG ringing.
- **Split Breakdown (80/20 Stratified):**
  - `data/splits/kvasir_train.json` (800 images: 53.4% Medium, 26.6% Large, 20.0% Small)
  - `data/splits/kvasir_val.json` (200 images: 53.5% Medium, 26.5% Large, 20.0% Small)

---

## 3. Out-of-Distribution: CVC-ClinicDB
- **Sample Count:** 612 frames and ground truth masks (`data/splits/cvc_clinicdb_test.json`).
- **Domain Shift Properties:**
  - Higher proportion of Small Polyps (<5% area): **37.42%** vs 20.00% in Kvasir-SEG.
  - Different camera sensors, optical illumination, and mucosal color distributions.
- **Isolation Protocol:** Strictly evaluated zero-shot with locked checkpoint weights.
