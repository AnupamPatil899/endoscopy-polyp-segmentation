# Clinical Failure Analysis & Diagnostic Taxonomy

**Analysis Notebook:** [`notebooks/failure_analysis.ipynb`](file:///home/anupa/polyp-segmentation/notebooks/failure_analysis.ipynb)  
**Study Cohort:** In-Distribution (Kvasir-SEG, Norway, N=200) vs Zero-Shot Out-of-Distribution (CVC-ClinicDB, Spain, N=612)

---

## 1. Failure Taxonomy & Root Causes

Exhaustive visual analysis of worst-case failures revealed 5 distinct root causes:

| Category | Typical Visual Signature | Primary Error Mode | Recommended Engineering Countermeasure |
|---|---|---|---|
| **1. Specular Glare & Reflections** | High-intensity white glare on wet mucosal peaks | False Positive peaks / Boundary truncation | Real-time specular glare suppression / inpainting filter in preprocessing pipeline |
| **2. Flat / Sessile Morphology (Paris IIa/IIb)** | Subtle mucosal elevation with no shadow margin | False Negative (Missed polyp tissue) | Active learning on flat adenoma cohorts + dual-threshold detection alarms ($\tau=0.35$) |
| **3. Motion & Focus Blur** | Washed-out pit pattern from rapid scope translation | Under-segmentation / Low confidence | Temporal frame aggregation / multi-frame optical flow consistency |
| **4. Low-Contrast Boundary Ambiguity** | Chromatic blending with surrounding mucosal folds | Boundary dilation / Over-segmentation | Multi-scale feature enhancement (e.g. boundary-aware loss or transformer backbone) |
| **5. Sub-1% Micro-Adenomas** | Diminutive lesions (<0.8% frame area) | False Negative (Missed lesion) | High-resolution patch-based inference or feature pyramid attention heads |

---

## 2. Visual Multi-View Verification

All failure modes are visualized with 5-panel comparative overlays (Raw RGB, Ground Truth Mask, Binary Prediction, Jet Probability Heatmap, and TP/FP/FN Error Maps) directly in the Jupyter Notebook: [`notebooks/failure_analysis.ipynb`](file:///home/anupa/polyp-segmentation/notebooks/failure_analysis.ipynb).

---

## 3. Engineering Recommendations for Clinical Deployment

1. **Dual Operating Thresholds:**
   - **Screening Alarm Mode ($\tau = 0.35$):** Prioritizes detection recall ($>92\%$) to minimize missed subtle lesions.
   - **Quantification Mode ($\tau = 0.50$):** Standard threshold for boundary quantification.
2. **Optical Glare Mitigation:**
   - Deploy real-time specular highlight detection and inpainting before neural net feature extraction.
3. **Morphology-Stratified Active Learning:**
   - Prioritize data collection on Paris Class IIa/IIb flat adenomas to address the tail risk.
