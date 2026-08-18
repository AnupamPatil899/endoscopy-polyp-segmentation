"""
src/infer.py - Modular Polyp Inference and Post-Processing Engine

Provides programmatic and CLI interfaces for single-image and batch polyp segmentation,
probability heatmap estimation, connected-component lesion counting, bounding box extraction,
and error map visualization against ground truth masks.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from src.dataset import get_val_transforms
from src.model import build_model


class PolypPredictor:
    """
    Production-ready inference wrapper for PolypUNet.
    Handles preprocessing, tensor conversion, probability prediction,
    thresholding, connected-component analysis, and visual overlay synthesis.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "outputs/checkpoints/baseline_kvasir_unet_resnet34_best.pth",
        encoder_name: str = "resnet34",
        img_size: int = 352,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.img_size = img_size
        self.checkpoint_path = Path(checkpoint_path)
        
        # Device resolution
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        # Build Model
        self.model = build_model(
            encoder_name=encoder_name,
            encoder_weights=None,
            device=self.device,
        )
        
        # Load weights
        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[PolypPredictor] Loaded weights from {self.checkpoint_path} to {self.device}")
        else:
            print(f"[PolypPredictor] Warning: Checkpoint not found at {self.checkpoint_path}. Using uninitialized weights.")
            self.model.eval()

        self.transform = get_val_transforms(img_size=(self.img_size, self.img_size))

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        threshold: float = 0.5,
        gt_mask_input: Optional[Union[str, Path, np.ndarray, Image.Image]] = None,
    ) -> Dict:
        """
        Executes complete inference pipeline on a single image.

        Args:
            image_input: Filepath, NumPy array (RGB), or PIL Image.
            threshold: Binary classification threshold (default: 0.50).
            gt_mask_input: Optional ground truth mask for metrics calculation.

        Returns:
            Dictionary containing raw image, probability map, binary mask,
            polyp count, bounding boxes, size bucket, metrics, and latency.
        """
        start_time = time.perf_counter()

        # 1. Load & Standardize RGB image
        if isinstance(image_input, (str, Path)):
            raw_bgr = cv2.imread(str(image_input))
            if raw_bgr is None:
                raise FileNotFoundError(f"Could not load image from: {image_input}")
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            raw_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            raw_rgb = image_input.copy()
            if raw_rgb.ndim == 2:
                raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_GRAY2RGB)
            elif raw_rgb.shape[2] == 4:
                raw_rgb = cv2.cvtColor(raw_rgb, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = raw_rgb.shape[:2]

        # 2. Transform & Preprocess
        augmented = self.transform(image=raw_rgb)
        tensor_img = augmented["image"].unsqueeze(0).to(self.device)

        # 3. Model Inference
        with torch.no_grad():
            probs_tensor = self.model.predict_proba(tensor_img)
            prob_map_352 = probs_tensor.squeeze().cpu().numpy().astype(np.float32)

        # Resize probability map back to original image resolution
        prob_map_orig = cv2.resize(prob_map_352, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        pred_mask_orig = (prob_map_orig >= threshold).astype(np.uint8)

        # 4. Connected Component Analysis (Polyp counting and BBoxes)
        labeled_pred, num_polyps = ndimage.label(pred_mask_orig)
        polyps_metadata = []
        total_pred_pixels = int(np.sum(pred_mask_orig == 1))
        area_pct = float((total_pred_pixels / (orig_h * orig_w)) * 100.0)

        for p_idx in range(1, num_polyps + 1):
            single_polyp = (labeled_pred == p_idx)
            polyp_pixels = int(np.sum(single_polyp))
            if polyp_pixels < 20:  # Noise filter: ignore <20px micro-artifacts
                continue
            
            y_indices, x_indices = np.where(single_polyp)
            ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
            xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
            
            polyps_metadata.append({
                "polyp_id": len(polyps_metadata) + 1,
                "bbox": [xmin, ymin, xmax, ymax],
                "area_pixels": polyp_pixels,
                "area_pct": float((polyp_pixels / (orig_h * orig_w)) * 100.0),
                "mean_confidence": float(np.mean(prob_map_orig[single_polyp])),
            })

        # Size classification
        if area_pct == 0:
            size_bucket = "None"
        elif area_pct < 5.0:
            size_bucket = "Small (<5%)"
        elif area_pct <= 20.0:
            size_bucket = "Medium (5-20%)"
        else:
            size_bucket = "Large (>20%)"

        # 5. Optional Ground Truth Mask Processing & Metrics
        gt_mask_orig = None
        metrics = None
        if gt_mask_input is not None:
            if isinstance(gt_mask_input, (str, Path)):
                raw_gt = cv2.imread(str(gt_mask_input), cv2.IMREAD_GRAYSCALE)
                if raw_gt is not None:
                    gt_mask_orig = (cv2.resize(raw_gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
            elif isinstance(gt_mask_input, Image.Image):
                gt_arr = np.array(gt_mask_input.convert("L"))
                gt_mask_orig = (cv2.resize(gt_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
            elif isinstance(gt_mask_input, np.ndarray):
                gt_arr = gt_mask_input.copy()
                if gt_arr.ndim == 3:
                    gt_arr = cv2.cvtColor(gt_arr, cv2.COLOR_RGB2GRAY)
                gt_mask_orig = (cv2.resize(gt_arr, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

            if gt_mask_orig is not None:
                intersection = np.sum((pred_mask_orig == 1) & (gt_mask_orig == 1))
                p_sum = np.sum(pred_mask_orig == 1)
                g_sum = np.sum(gt_mask_orig == 1)
                dice = float((2.0 * intersection) / (p_sum + g_sum + 1e-7)) if (p_sum + g_sum) > 0 else 1.0
                iou = float(intersection / (p_sum + g_sum - intersection + 1e-7)) if (p_sum + g_sum) > 0 else 1.0
                precision = float(intersection / (p_sum + 1e-7)) if p_sum > 0 else 1.0
                recall = float(intersection / (g_sum + 1e-7)) if g_sum > 0 else 1.0

                metrics = {
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                }

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "image_rgb": raw_rgb,
            "prob_map": prob_map_orig,
            "pred_mask": pred_mask_orig,
            "gt_mask": gt_mask_orig,
            "polyp_count": len(polyps_metadata),
            "polyps_metadata": polyps_metadata,
            "area_pct": area_pct,
            "size_bucket": size_bucket,
            "metrics": metrics,
            "threshold": threshold,
            "latency_ms": latency_ms,
            "resolution": (orig_w, orig_h),
        }

    def render_overlay(
        self,
        image_rgb: np.ndarray,
        pred_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 230, 0),
        alpha: float = 0.45,
        draw_contours: bool = True,
        draw_bboxes: bool = True,
        polyps_metadata: Optional[List[Dict]] = None,
    ) -> np.ndarray:
        """Renders an alpha-blended mask overlay on the RGB image with optional contours & bounding boxes."""
        overlay = image_rgb.copy()
        mask_bool = (pred_mask == 1)

        if np.any(mask_bool):
            color_mask = np.zeros_like(image_rgb)
            color_mask[mask_bool] = color
            overlay[mask_bool] = cv2.addWeighted(image_rgb[mask_bool], 1.0 - alpha, color_mask[mask_bool], alpha, 0)

        if draw_contours:
            contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        if draw_bboxes and polyps_metadata:
            for p in polyps_metadata:
                xmin, ymin, xmax, ymax = p["bbox"]
                cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 220, 0), 2)
                label = f"Polyp #{p['polyp_id']} ({p['mean_confidence']*100:.0f}%)"
                cv2.putText(overlay, label, (xmin, max(ymin - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

        return overlay

    def render_heatmap(
        self,
        image_rgb: np.ndarray,
        prob_map: np.ndarray,
        colormap: str = "JET",
        alpha: float = 0.50,
    ) -> np.ndarray:
        """Renders a continuous colormap probability heatmap overlaid onto the colonoscopy image."""
        cm_map = {
            "JET": cv2.COLORMAP_JET,
            "TURBO": cv2.COLORMAP_TURBO,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "HOT": cv2.COLORMAP_HOT,
        }
        selected_cm = cm_map.get(colormap.upper(), cv2.COLORMAP_JET)

        prob_uint8 = np.uint8(255 * np.clip(prob_map, 0.0, 1.0))
        heatmap_bgr = cv2.applyColorMap(prob_uint8, selected_cm)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)
        return blended

    def render_error_map(
        self,
        image_rgb: np.ndarray,
        pred_mask: np.ndarray,
        gt_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Renders a 3-color diagnostic clinical error map:
        - Green: True Positives (Accurate overlap)
        - Blue: False Positives (Over-segmentation)
        - Red: False Negatives (Missed polyp tissue)
        """
        tp = (pred_mask == 1) & (gt_mask == 1)
        fp = (pred_mask == 1) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask == 1)

        error_vis = image_rgb.copy()
        overlay = np.zeros_like(image_rgb)
        overlay[tp] = [0, 230, 0]    # Green TP
        overlay[fp] = [30, 120, 255] # Blue FP
        overlay[fn] = [255, 30, 30]  # Red FN

        active = tp | fp | fn
        if np.any(active):
            error_vis[active] = cv2.addWeighted(error_vis[active], 0.35, overlay[active], 0.65, 0)

        # Contours: White for GT, Yellow for Prediction
        contours_gt, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_pred, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(error_vis, contours_gt, -1, (255, 255, 255), 2)
        cv2.drawContours(error_vis, contours_pred, -1, (255, 255, 0), 2)

        return error_vis


def main():
    parser = argparse.ArgumentParser(description="Clinical Polyp Segmentation Inference Engine")
    parser.add_argument("--image", type=str, required=True, help="Path to input endoscopic image")
    parser.add_argument("--mask", type=str, default=None, help="Optional path to ground truth mask")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/baseline_kvasir_unet_resnet34_best.pth")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--output_dir", type=str, default="outputs/inference_results", help="Output directory")
    args = parser.parse_args()

    predictor = PolypPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict(image_input=args.image, threshold=args.threshold, gt_mask_input=args.mask)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.image).stem
    overlay_img = predictor.render_overlay(res["image_rgb"], res["pred_mask"], polyps_metadata=res["polyps_metadata"])
    heatmap_img = predictor.render_heatmap(res["image_rgb"], res["prob_map"])

    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / f"{stem}_heatmap.png"), cv2.cvtColor(heatmap_img, cv2.COLOR_RGB2BGR))

    if res["gt_mask"] is not None:
        error_img = predictor.render_error_map(res["image_rgb"], res["pred_mask"], res["gt_mask"])
        cv2.imwrite(str(out_dir / f"{stem}_error_map.png"), cv2.cvtColor(error_img, cv2.COLOR_RGB2BGR))

    print("\n" + "="*50)
    print(f"  Inference Result for: {Path(args.image).name}")
    print("="*50)
    print(f"  Detected Polyps:  {res['polyp_count']}")
    print(f"  Lesion Area %:    {res['area_pct']:.2f}% ({res['size_bucket']})")
    print(f"  Latency:          {res['latency_ms']:.2f} ms")
    if res["metrics"]:
        print(f"  Dice Score:       {res['metrics']['dice']:.4f}")
        print(f"  IoU (Jaccard):    {res['metrics']['iou']:.4f}")
        print(f"  Precision:        {res['metrics']['precision']:.4f}")
        print(f"  Recall:           {res['metrics']['recall']:.4f}")
    print(f"  Saved artifacts to: {out_dir}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
