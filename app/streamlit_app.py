"""
app/streamlit_app.py - Interactive Clinical Polyp Segmentation Web Application

A clinical-grade diagnostic interface for real-time endoscopic polyp segmentation,
probability calibration visualization, lesion quantification, and multi-tier evaluation.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from src.infer import PolypPredictor


# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="EndoSeg AI — Clinical Polyp Segmentation",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .clinical-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-blue { background-color: #DBEAFE; color: #1E40AF; }
    .badge-green { background-color: #DCFCE7; color: #166534; }
    .badge-amber { background-color: #FEF3C7; color: #92400E; }
    .badge-purple { background-color: #F3E8FF; color: #6B21A8; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model Initialization
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ResNet34 U-Net Weights...")
def load_predictor() -> PolypPredictor:
    ckpt_path = project_root / "outputs/checkpoints/baseline_kvasir_unet_resnet34_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = PolypPredictor(checkpoint_path=ckpt_path, device=device)
    return predictor

predictor = load_predictor()


# -----------------------------------------------------------------------------
# Curated Preset Samples
# -----------------------------------------------------------------------------
PRESETS = {
    "Select a preset case...": None,
    "In-Distribution Standard (Kvasir-SEG)": {
        "image": project_root / "Data/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg",
        "mask": project_root / "Data/Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg",
        "desc": "In-Distribution validation sample from Kvasir-SEG (Norway). High-contrast elevated polyp.",
        "origin": "Kvasir-SEG (Norway)",
    },
    "OOD Small Polyp (CVC-ClinicDB #100)": {
        "image": project_root / "Data/CVC-ClinicDB/images/100.png",
        "mask": project_root / "Data/CVC-ClinicDB/masks/100.png",
        "desc": "Zero-shot out-of-distribution sample from CVC-ClinicDB (Spain). Small 6.6% lesion.",
        "origin": "CVC-ClinicDB (Spain)",
    },
    "OOD Specular Reflection Glare (CVC-ClinicDB #1)": {
        "image": project_root / "Data/CVC-ClinicDB/images/1.png",
        "mask": project_root / "Data/CVC-ClinicDB/masks/1.png",
        "desc": "Challenging OOD sample with heavy specular mucosal reflection and highlight saturation.",
        "origin": "CVC-ClinicDB (Spain)",
    },
    "OOD Motion & Focus Blur (CVC-ClinicDB #447)": {
        "image": project_root / "Data/CVC-ClinicDB/images/447.png",
        "mask": project_root / "Data/CVC-ClinicDB/masks/447.png",
        "desc": "OOD colonoscopy video frame with lateral scope translation motion blur.",
        "origin": "CVC-ClinicDB (Spain)",
    },
    "In-Distribution Flat Lesion (Kvasir-SEG #5wph)": {
        "image": project_root / "Data/Kvasir-SEG/images/cju5wphwwlu3m0987hh3ltg88.jpg",
        "mask": project_root / "Data/Kvasir-SEG/masks/cju5wphwwlu3m0987hh3ltg88.jpg",
        "desc": "Flat sessile morphology (Paris Class IIa/IIb) with subtle mucosal elevation.",
        "origin": "Kvasir-SEG (Norway)",
    },
}

COLOR_MAP = {
    "Clinical Green (Default)": (0, 230, 0),
    "Electric Cyan": (0, 235, 235),
    "Vibrant Amber": (255, 175, 0),
    "Coral Red": (255, 50, 50),
    "Neon Purple": (180, 50, 255),
}


# -----------------------------------------------------------------------------
# Sidebar: Diagnostic & Rendering Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=64)
    st.title("Control Center")
    st.markdown("---")

    st.subheader("1. Sample Source")
    input_mode = st.radio("Input Mode", ["Curated Demo Presets", "Upload Custom Image"], index=0)

    selected_image_path = None
    selected_mask_path = None
    uploaded_image = None
    uploaded_mask = None

    if input_mode == "Curated Demo Presets":
        preset_choice = st.selectbox("Choose Case", list(PRESETS.keys()), index=1)
        if PRESETS[preset_choice] is not None:
            selected_image_path = PRESETS[preset_choice]["image"]
            selected_mask_path = PRESETS[preset_choice]["mask"]
            st.info(f"**Origin:** {PRESETS[preset_choice]['origin']}\n\n{PRESETS[preset_choice]['desc']}")
    else:
        uploaded_image = st.file_uploader("Upload Colonoscopy Image (JPG/PNG)", type=["jpg", "jpeg", "png", "webp"])
        uploaded_mask = st.file_uploader("Upload Ground Truth Mask (Optional)", type=["jpg", "jpeg", "png"])

    st.markdown("---")
    st.subheader("2. Segmentation Tuning")

    threshold = st.slider(
        "Decision Threshold (τ)",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Operating cutoff probability for binary polyp classification. Lower values (e.g. 0.35) increase sensitivity for subtle flat adenomas; higher values reduce false alarms.",
    )

    mask_alpha = st.slider(
        "Mask Opacity (Alpha)",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )

    selected_color_name = st.selectbox("Mask Color Palette", list(COLOR_MAP.keys()), index=0)
    selected_rgb = COLOR_MAP[selected_color_name]

    st.markdown("---")
    st.subheader("3. Visualization Options")
    heatmap_colormap = st.selectbox("Heatmap Palette", ["JET", "TURBO", "VIRIDIS", "INFERNO", "HOT"], index=0)
    show_bboxes = st.checkbox("Draw Polyp Bounding Boxes", value=True)
    show_contours = st.checkbox("Draw Boundary Contours", value=True)


# -----------------------------------------------------------------------------
# Main Application Content
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🔬 Clinical Polyp Segmentation & Diagnostic Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time deep learning inference, probability calibration, and multi-tier distribution shift assessment on colonoscopy imagery.</div>', unsafe_allow_html=True)

# System Architecture & Benchmark Badges
st.markdown("""
<span class="clinical-badge badge-blue">Model: ResNet34 U-Net</span>
<span class="clinical-badge badge-green">In-Dist Val Dice: 90.50%</span>
<span class="clinical-badge badge-amber">OOD Zero-Shot Dice: 85.59%</span>
<span class="clinical-badge badge-purple">Small Polyp Recall: 90.00%</span>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs: Interactive Visualizer vs Study Summary
tab_inference, tab_insights = st.tabs(["🚀 Live Diagnostic Visualizer", "📊 Clinical Generalization Study & Findings"])


# -----------------------------------------------------------------------------
# TAB 1: Live Diagnostic Visualizer
# -----------------------------------------------------------------------------
with tab_inference:
    # Prepare inputs
    img_input = None
    mask_input = None

    if input_mode == "Curated Demo Presets" and selected_image_path is not None:
        img_input = selected_image_path
        mask_input = selected_mask_path
    elif input_mode == "Upload Custom Image" and uploaded_image is not None:
        img_input = Image.open(uploaded_image)
        if uploaded_mask is not None:
            mask_input = Image.open(uploaded_mask)

    if img_input is not None:
        # Run Inference
        with st.spinner("Processing colonoscopy frame..."):
            res = predictor.predict(
                image_input=img_input,
                threshold=threshold,
                gt_mask_input=mask_input,
            )

        # Synthesize Views
        overlay_view = predictor.render_overlay(
            image_rgb=res["image_rgb"],
            pred_mask=res["pred_mask"],
            color=selected_rgb,
            alpha=mask_alpha,
            draw_contours=show_contours,
            draw_bboxes=show_bboxes,
            polyps_metadata=res["polyps_metadata"],
        )

        heatmap_view = predictor.render_heatmap(
            image_rgb=res["image_rgb"],
            prob_map=res["prob_map"],
            colormap=heatmap_colormap,
            alpha=0.50,
        )

        error_view = None
        if res["gt_mask"] is not None:
            error_view = predictor.render_error_map(
                image_rgb=res["image_rgb"],
                pred_mask=res["pred_mask"],
                gt_mask=res["gt_mask"],
            )

        # --- Clinical Telemetry Dashboard ---
        st.subheader("Clinical Telemetry & Lesion Biomarkers")
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        with col_m1:
            st.metric("Detected Polyps", f"{res['polyp_count']}", delta="Connected Components")
        with col_m2:
            st.metric("Lesion Area %", f"{res['area_pct']:.2f}%", delta=res["size_bucket"])
        with col_m3:
            mean_conf = np.mean(res["prob_map"][res["pred_mask"] == 1]) if np.any(res["pred_mask"] == 1) else 0.0
            st.metric("Mean Confidence", f"{mean_conf*100:.1f}%", delta="Lesion Region")
        with col_m4:
            st.metric("Inference Latency", f"{res['latency_ms']:.1f} ms", delta=f"{predictor.device.type.upper()}")
        with col_m5:
            if res["metrics"]:
                st.metric("Dice Overlap", f"{res['metrics']['dice']*100:.1f}%", delta=f"IoU: {res['metrics']['iou']*100:.1f}%")
            else:
                st.metric("Ground Truth", "Not Provided", delta="Upload to evaluate")

        # Ground Truth Metrics Detail (if available)
        if res["metrics"]:
            st.success(
                f"**Ground Truth Validation:** "
                f"Dice Score: **{res['metrics']['dice']:.4f}** ({res['metrics']['dice']*100:.2f}%) | "
                f"IoU (Jaccard): **{res['metrics']['iou']:.4f}** | "
                f"Pixel Precision: **{res['metrics']['precision']:.4f}** | "
                f"Pixel Recall: **{res['metrics']['recall']:.4f}**"
            )

        st.markdown("---")

        # --- Multi-View Image Inspection Grid ---
        st.subheader("Multi-View Optical Inspection")

        if error_view is not None:
            grid_cols = st.columns(4)
            with grid_cols[0]:
                st.markdown("**1. Raw Endoscopy Frame**")
                st.image(res["image_rgb"], width="stretch")
            with grid_cols[1]:
                st.markdown(f"**2. AI Segmentation (τ={threshold:.2f})**")
                st.image(overlay_view, width="stretch")
            with grid_cols[2]:
                st.markdown(f"**3. Probability Heatmap ({heatmap_colormap})**")
                st.image(heatmap_view, width="stretch")
            with grid_cols[3]:
                st.markdown("**4. Diagnostic Error Map**")
                st.image(error_view, width="stretch")
                st.caption("🟢 Green: True Positives | 🔵 Blue: False Positives | 🔴 Red: False Negatives")
        else:
            grid_cols = st.columns(3)
            with grid_cols[0]:
                st.markdown("**1. Raw Endoscopy Frame**")
                st.image(res["image_rgb"], width="stretch")
            with grid_cols[1]:
                st.markdown(f"**2. AI Segmentation (τ={threshold:.2f})**")
                st.image(overlay_view, width="stretch")
            with grid_cols[2]:
                st.markdown(f"**3. Probability Heatmap ({heatmap_colormap})**")
                st.image(heatmap_view, width="stretch")

        # Detailed Polyp Entity Breakdown
        if res["polyps_metadata"]:
            with st.expander("🔍 Detailed Connected-Component Entity Metadata", expanded=False):
                df_polyps = pd.DataFrame(res["polyps_metadata"])
                df_polyps["bbox"] = df_polyps["bbox"].apply(lambda b: f"[{b[0]}, {b[1]}, {b[2]}, {b[3]}]")
                df_polyps["area_pct"] = df_polyps["area_pct"].apply(lambda a: f"{a:.3f}%")
                df_polyps["mean_confidence"] = df_polyps["mean_confidence"].apply(lambda c: f"{c*100:.2f}%")
                st.dataframe(df_polyps, width="stretch")

    else:
        st.info("👈 Please select a curated demo case or upload a colonoscopy image from the sidebar to begin inference.")


# -----------------------------------------------------------------------------
# TAB 2: Clinical Generalization Study & Findings
# -----------------------------------------------------------------------------
with tab_insights:
    st.subheader("Distribution Shift & Generalization Study Overview")

    st.markdown("""
    This clinical benchmark investigates model generalization when transferred across independent hospital centers with zero fine-tuning:
    - **Training Source (In-Distribution):** **Kvasir-SEG** (Vestre Viken Hospital Trust, Norway, N=1,000)
    - **Held-Out Test (Out-of-Distribution):** **CVC-ClinicDB** (Hospital Clinic Barcelona, Spain, N=612)
    """)

    # Side-by-Side Performance Table
    st.markdown("### 1. Multi-Tier Benchmark Comparison")
    eval_table = pd.DataFrame([
        {"Metric": "Mean Dice Score", "In-Distribution (Kvasir)": "0.9050 (90.50%)", "Out-of-Distribution (CVC)": "0.8559 (85.59%)", "Generalization Delta": "-5.43%"},
        {"Metric": "Median Dice Score", "In-Distribution (Kvasir)": "0.9569 (95.69%)", "Out-of-Distribution (CVC)": "0.9263 (92.63%)", "Generalization Delta": "-3.20%"},
        {"Metric": "Worst-Decile (P10) Dice", "In-Distribution (Kvasir)": "0.7688 (76.88%)", "Out-of-Distribution (CVC)": "0.6641 (66.41%)", "Generalization Delta": "-13.62% (Tail Drop)"},
        {"Metric": "Mean IoU (Jaccard)", "In-Distribution (Kvasir)": "0.8480 (84.80%)", "Out-of-Distribution (CVC)": "0.7816 (78.16%)", "Generalization Delta": "-7.84%"},
        {"Metric": "Polyp Detection Sensitivity", "In-Distribution (Kvasir)": "87.93% (204/232)", "Out-of-Distribution (CVC)": "88.43% (627/709)", "Generalization Delta": "+0.50%"},
        {"Metric": "Small Polyp Sensitivity (<5%)", "In-Distribution (Kvasir)": "93.02% (40/43)", "Out-of-Distribution (CVC)": "90.00% (234/260)", "Generalization Delta": "-3.02%"},
        {"Metric": "Expected Calibration Error (ECE)", "In-Distribution (Kvasir)": "0.0157", "Out-of-Distribution (CVC)": "0.0179", "Generalization Delta": "+0.0022"},
        {"Metric": "Brier Score", "In-Distribution (Kvasir)": "0.0209", "Out-of-Distribution (CVC)": "0.0210", "Generalization Delta": "+0.0001"},
    ])
    st.table(eval_table)

    st.markdown("---")

    # Key Discoveries
    col_k1, col_k2 = st.columns(2)
    with col_k1:
        st.markdown("### 2. The 'Tail Drop' Discovery")
        st.markdown("""
        - In medical deep learning, aggregate mean metrics hide catastrophic edge-case failures.
        - While **median Dice dropped by only 3.20%**, the **10th percentile (P10) dropped by 13.62%**.
        - Distribution shift primarily expands the tail of difficult morphologies rather than shifting the main mode.
        """)

    with col_k2:
        st.markdown("### 3. Small Polyp Resilience")
        st.markdown(r"""
        - Small adenomas ($<5\%$ area) represent **37.4%** of CVC-ClinicDB vs **20.0%** in Kvasir-SEG.
        - Despite camera sensor differences, the model retained **90.00% clinical detection sensitivity** on small polyps.
        - Domain-specific augmentations (specular glare simulation, motion blur) regularized the encoder effectively.
        """)

    st.markdown("---")

    # Failure Taxonomy
    st.markdown("### 4. Five Clinical Failure Categories")
    st.markdown("""
    1. **Specular Glare & Reflection:** Optical blowout on wet mucosa causing pseudo-polyp peaks or border truncation.
    2. **Flat / Sessile Morphology (Paris Class IIa/IIb):** Minimal elevation lacking shadow margins, blending with colon walls.
    3. **Motion & Focus Blur:** Fast scope translation blurring mucosal pit textures.
    4. **Low-Contrast Boundary Ambiguity:** Lesion hues blending with background folds.
    5. **Sub-1% Micro-Polyps:** Scale mismatch causing feature vanishing during multi-scale pooling.
    """)
