# 🚀 Implementation Plan: Improving DDR OSR Performance

**Goal**: Improve Known Class Accuracy (specifically Mild/Severe) while maintaining >99% AUROC for Unknown Detection.

## 1. Diagnosis Summary
- **Current Status**: Strong OSR (99.42% AUROC), Weak Fine-grained Classification (Mild 11%, Severe 46%).
- **Root Cause**: ResNet50 struggles with subtle, fine-grained features (microaneurysms, specific hemorrhage counts) needed for Mild/Severe differentiation.
- **Constraint**: Any change must **not** degrade the feature space separation that enables high OSR performance.

## 2. Proposed Changes

### Phase 1: Robust Inference (Low Risk)
**Objective**: Boost performance without retraining.
- **Technique**: **Test Time Augmentation (TTA)**.
- **Method**: Average predictions across 5 augmented versions of each test image (HorizontalFlip, VerticalFlip, slight Rotation).
- **Expected Impact**: +1-3% overall accuracy, potential stabilization of Mild predictions.
- **Risk**: None (Inference only).

### Phase 2: Model Enhancement (Medium Risk)
**Objective**: Improve fine-grained feature extraction for Severe/Mild classes.
- **Technique**: **CBAM (Convolutional Block Attention Module)**.
- **Method**: Integrate Channel and Spatial Attention modules into the ResNet50 bottleneck blocks.
- **Rationale**: Helps model focus on small lesions (microaneurysms) rather than background noise.
- **Expected Impact**: Significant improvement in Severe class (counting features).
- **Risk**: Low. Attention usually improves feature discriminability, which helps OSR.

### Phase 3: Training Stabilization (Medium Risk)
**Objective**: Prevent overconfidence in majority classes (No_DR) and improve generalization.
- **Technique**: **Label Smoothing (ε=0.1)**.
- **Method**: Replace hard 0/1 targets with soft targets (0.9 for true class, 0.025 for others).
- **Rationale**: Prevents the model from becoming too confident on "easy" No_DR samples, potentially encouraging it to learn more robust features for "hard" Mild samples.
- **Expected Impact**: Better calibration, potentially better OSR scores.
- **Risk**: Low. Generally helps OSR.

## 3. Implementation Steps

### Step 1: Implement TTA
- [ ] Create `evaluate_tta.py`.
- [ ] Implement 5-crop/flip averaging.
- [ ] Evaluate on Test Set and Cross-Dataset (OSR).

### Step 2: Integrate CBAM
- [ ] Modify `models/resnet_cbam.py` (or similar).
- [ ] Add CBAM blocks to ResNet architecture.
- [ ] Retrain with best hyperparameters (80/10/10 split, Focal Loss γ=3.0).

### Step 3: Label Smoothing
- [ ] Modify `train_focal_loss_enhanced.py` to support Label Smoothing Cross Entropy (or combine with Focal Loss).
- [ ] Retrain and compare.

## 4. Verification Plan
For each change:
1. **Check Closed-Set Accuracy**: Must improve (or at least maintain) overall accuracy > 81%.
2. **Check Per-Class Accuracy**: Monitor Mild/Severe specifically.
3. **Check OSR AUROC**: **MUST remain > 99%**.

## 5. Timeline
- **TTA**: ~1 hour (Coding + Eval)
- **CBAM**: ~2-3 hours (Coding + Training)
- **Label Smoothing**: ~2 hours (Coding + Training)
