# Clinical & Out-of-Distribution Evaluation Report

## 1. Executive Benchmark Summary

This report documents the empirical evaluation of the ResNet34 U-Net model trained strictly on **Kvasir-SEG** (Vestre Viken Hospital Trust, Norway) and evaluated zero-shot on **CVC-ClinicDB** (Hospital Clinic Barcelona, Spain).

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
| Polyp-Level Sensitivity   | 0.8793 (87.93%)       | 0.8843 (88.43%)       | +0.0050 (+0.57%)  |
| Expected Calib. Error(ECE)| 0.0157                | 0.0179                | +0.0022           |
| Brier Score               | 0.0209                | 0.0210                | +0.0001           |
+---------------------------+-----------------------+-----------------------+-------------------+
```

---

## 2. Size Stratified Performance Analysis

```
+---------------+-------------------+-----------------------+-----------------------+-------------------+
| Size Bucket   | Metric            | Kvasir-SEG (In-Dist)  | CVC-ClinicDB (OOD)    | Generalization    |
|               |                   |                       |                       | Delta             |
+---------------+-------------------+-----------------------+-----------------------+-------------------+
| Small (<5%)   | Sample Count      | 40 samples (20.0%)    | 229 samples (37.4%)   | +17.4% population |
|               | Mean Dice         | 0.8646                | 0.8308                | -0.0338           |
|               | Median Dice       | 0.9312                | 0.9203                | -0.0109           |
|               | Worst-Decile (P10)| 0.6576                | 0.6262                | -0.0314           |
|               | Polyp Sensitivity | 93.02% (40/43 polyps) | 90.00% (234/260 polyps)| -3.02%           |
+---------------+-------------------+-----------------------+-----------------------+-------------------+
| Medium (5-20%)| Sample Count      | 107 samples (53.5%)   | 321 samples (52.5%)   | -1.0% population  |
|               | Mean Dice         | 0.9244                | 0.8801                | -0.0443           |
|               | Median Dice       | 0.9618                | 0.9381                | -0.0237           |
|               | Worst-Decile (P10)| 0.8127                | 0.7144                | -0.0983           |
|               | Polyp Sensitivity | 92.50% (111/120)      | 90.24% (333/369)      | -2.26%            |
+---------------+-------------------+-----------------------+-----------------------+-------------------+
| Large (>20%)  | Sample Count      | 53 samples (26.5%)    | 62 samples (10.1%)    | -16.4% population |
|               | Mean Dice         | 0.8963                | 0.8229                | -0.0734           |
|               | Median Dice       | 0.9653                | 0.8988                | -0.0665           |
|               | Worst-Decile (P10)| 0.7547                | 0.5777                | -0.1770           |
|               | Polyp Sensitivity | 76.81% (53/69)        | 75.00% (60/80)        | -1.81%            |
+---------------+-------------------+-----------------------+-----------------------+-------------------+
```

---

## 3. Key Findings

1. **Robust Clinical Sensitivity Across Centers**:
   - Despite camera hardware shift (Olympus Lucera in Kvasir vs. Olympus Q160AL in CVC-ClinicDB), the model maintained **88.43% polyp detection sensitivity** without fine-tuning.
   - For **small polyps**, detection sensitivity remained at **90.00%**, demonstrating that domain augmentations helped preserve invariant lesion features.

2. **The "Tail Drop" Effect**:
   - While median Dice dropped by only **3.20%** (0.9569 -> 0.9263), the 10th percentile (P10 worst cases) dropped by **13.62%** (0.7688 -> 0.6641).
   - This illustrates why medical AI benchmarks must report percentile distributions (median, P10) rather than solely mean metrics.

3. **Calibration Under Domain Shift**:
   - Expected Calibration Error (ECE) rose slightly from **0.0157** to **0.0179**, and Brier score remained low at **0.0210**.
   - Predictions outputting $>0.90$ probability had $>0.94$ empirical pixel accuracy, confirming the model's confidence scores can be trusted for clinical triaging.
