# OSR Strategy Pivot: Focal Loss + Mahalanobis (FINAL)

## 1. Final Strategy
We have reverted to the **proven winning strategy** that achieved the target performance:
**Focal Loss Classifier + Mahalanobis Distance Scoring.**

### Why this choice?
1.  **Proven Success:** This configuration previously achieved **86.23% Known Accuracy** and **~85% AUROC**.
2.  **Simplicity:** It avoids the complexity and instability of Outlier Exposure (OE) and LogitNorm, which caused class collapse and accuracy degradation.
3.  **Robustness:** Focal Loss handles the class imbalance (Class 1) effectively, while Mahalanobis Distance handles the OOD detection in feature space.

## 2. Implementation Details
*   **Training (`train_focal_energy.py`):**
    *   **Loss:** Focal Loss (`gamma=2.0`, `alpha=[0.4, 1.0, 0.6]`).
    *   **Optimizer:** AdamW with Differential Learning Rates.
    *   **Epochs:** 40.
*   **Evaluation (`mahalanobis_ood.py`):**
    *   **Features:** Penultimate layer (2048-dim).
    *   **Score:** Minimum Mahalanobis distance to class means.
    *   **Regularization:** `epsilon=0.01`.

## 3. Results History
*   **ARPL (Buggy):** ~36% AUROC
*   **LogitNorm + OE:** ~60% Acc (Failed)
*   **Softmax + OE:** ~57% Acc (Failed)
*   **Focal Loss + Mahalanobis (FINAL RUN):**
    *   **Known Accuracy:** **86.93%** (Class 1: 82.70%!)
    *   **AUROC:** **83.86%**
    *   **Combined Score:** **85.37%**

## 4. Conclusion
The experimentation with OE and LogitNorm confirmed that the simpler, feature-based Mahalanobis approach is superior for this dataset's continuous severity shifts.

## 5. Artifacts
*   **Model Checkpoint:** `checkpoints/baseline_focal_mahalanobis.pth`
*   **Features & Stats:** `saved_features_baseline/`
    *   `train_features.npy`, `train_labels.npy` (Training data embeddings)
    *   `test_features.npy`, `test_labels.npy` (Test data embeddings)
    *   `class_means.npy`, `precision.npy` (Mahalanobis statistics)
