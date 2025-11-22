# Implementation Plan - Cross-Dataset OSR (DDR vs. ACRIMA)

## Goal
Train a model on the full **DDR dataset (5 classes)** to recognize all stages of Diabetic Retinopathy, and use the **ACRIMA dataset** (Glaucoma) as the "Unknown" class for Open Set Recognition.

## User Review Required
> [!IMPORTANT]
> **Dataset Action:** Please download the **ACRIMA** dataset.
> **Download Link:** [Figshare (Direct Link)](https://figshare.com/s/c2d31f850af14c5b5232) or [Kaggle](https://www.kaggle.com/datasets/felipekitamura/acrima).
> **Destination:** Please extract it to: `e:\Open-Set-Recognition-master\datasets\ACRIMA`
> **Expected Structure:**
> ```
> datasets/
>   ACRIMA/
>     Images/
>       Glaucoma/
>       Normal/
> ```

## Proposed Changes

### 1. Data Preparation
#### [NEW] `datasets/acrima.py`
- Create `AcrimDataset` class.
- Load images from `datasets/ACRIMA/Images/Glaucoma` (we only need the Glaucoma ones for "Unknowns").
- Assign label `5` (Unknown).
- Apply `transforms.Resize((224, 224))` and normalization.

### 2. Model Modification
#### [NEW] `train_full_ddr.py`
- Based on `train_oe_mahalanobis.py`.
- **Change:** `num_classes = 5` (No_DR, Mild, Moderate, Severe, Proliferative).
- **Change:** Use `train_class_num=5` for DDR loader.
- **Output:** Save to `checkpoints/resnet50_full_5class.pth`.

### 3. Evaluation Pipeline
#### [NEW] `evaluate_cross_dataset.py`
- Load `resnet50_full_5class.pth`.
- **Step 1:** Extract features from DDR Test Set (All 5 classes) -> These are "Knowns".
- **Step 2:** Extract features from ACRIMA (Glaucoma) -> These are "Unknowns".
- **Step 3:** Compute Mahalanobis Distance.
- **Step 4:** Calculate AUROC (DDR vs ACRIMA).

## Verification Plan

### Automated Tests
- **Dataset Check:** Script to verify `datasets/ACRIMA` exists and contains images.
- **Model Check:** Verify model outputs 5 scores.

### Manual Verification
- **AUROC Target:** We hope for > 85% AUROC.
- **Sanity Check:** Ensure the model doesn't classify Glaucoma as "Mild DR" (it should have high Mahalanobis distance).
