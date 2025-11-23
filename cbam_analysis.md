# CBAM Training Results Analysis

## Executive Summary
**Goal**: Improve Mild/Severe class accuracy while maintaining >99% AUROC for unknown detection.

**Result**: ❌ **Goal NOT Achieved** - CBAM did not improve fine-grained classification and slightly degraded OSR performance.

---

## Detailed Comparison

### 1. Closed-Set Classification Performance

| Metric | Baseline (ResNet50) | CBAM (ResNet50-CBAM) | Change |
|--------|---------------------|----------------------|--------|
| **Overall Accuracy** | 86.23% | 80.29% | **-5.94%** ❌ |
| **No_DR** | 91.53% | 92.03% | +0.50% ✓ |
| **Mild** | 83.33% | 15.87% | **-67.46%** ❌❌❌ |
| **Moderate** | 79.23% | 77.23% | -2.00% ⚠️ |
| **Severe** | N/A* | 45.83% | N/A |
| **Proliferative** | N/A* | 68.13% | N/A |

*Note: Baseline used 3-class setup (No_DR, Mild, Moderate), while CBAM used 5-class setup.

### 2. Open Set Recognition (OSR) Performance

| Metric | Baseline | CBAM | Change |
|--------|----------|------|--------|
| **AUROC** | **99.42%** | 97.72% | **-1.70%** ❌ |
| **Goal** | >99% | >99% | Failed |

---

## Key Insights

### 🔍 What Went Wrong?

1. **Catastrophic Mild Class Collapse (15.87%)**
   - CBAM was supposed to help with fine-grained features
   - Instead, Mild class accuracy **dropped by 67%**
   - This is the **opposite** of the intended effect

2. **OSR Performance Degradation**
   - AUROC dropped from 99.42% to 97.72%
   - Failed to meet the >99% constraint
   - Attention mechanism may have disrupted feature space separation

3. **Training Instability**
   - Training stopped at epoch 43 (early stopping triggered)
   - Best validation accuracy: 82.36%
   - Suggests the model struggled to converge effectively

### 🤔 Why Did CBAM Fail?

**Hypothesis 1: Attention Misdirection**
- CBAM's spatial attention may have focused on **irrelevant** regions (e.g., blood vessels, optic disc) instead of subtle lesions (microaneurysms)
- Without explicit supervision, attention can learn spurious correlations

**Hypothesis 2: Feature Space Disruption**
- The baseline ResNet50 (with Focal Loss) had learned a well-separated feature space
- Adding CBAM changed the feature geometry, breaking the Mahalanobis distance-based OOD detection

**Hypothesis 3: Oversampling Mismatch**
- Aggressive oversampling (Mild 20x, Severe 30x) may have caused the model to **overfit** to augmented versions of minority classes
- CBAM's attention might amplify this overfitting by focusing too narrowly

**Hypothesis 4: Insufficient Pretraining**
- Only 267/368 layers loaded from ImageNet (CBAM layers initialized randomly)
- CBAM modules may need more epochs to learn meaningful attention patterns

---

## Comparison with Original Goal

### Original Plan Expectations
- **Phase 1 (TTA)**: +1-3% accuracy, stabilize Mild predictions
  - **Result**: 81.72% accuracy, Mild 11.11% (minimal improvement)
  
- **Phase 2 (CBAM)**: Significant improvement in Severe/Mild classes
  - **Result**: Severe 45.83% (decent), Mild 15.87% (catastrophic failure)

- **Constraint**: Maintain >99% AUROC
  - **Result**: 97.72% AUROC ❌ (failed constraint)

---

## Recommendations

### ✅ What to Do Next

1. **Revert to Baseline**
   - The baseline ResNet50 + Focal Loss + Mahalanobis is superior
   - 86.23% accuracy, 99.42% AUROC meets the OSR constraint

2. **Alternative Approaches for Mild/Severe Improvement**
   - **Option A: Grad-CAM Analysis**
     - Visualize what the baseline model is looking at
     - Identify if it's missing key features for Mild/Severe
   
   - **Option B: Ensemble Methods**
     - Train separate models for different class pairs
     - Use ensemble voting for final prediction
   
   - **Option C: Data-Centric Approach**
     - Collect more Mild/Severe samples
     - Use expert annotations to highlight key regions (if available)
   
   - **Option D: Hierarchical Classification**
     - Stage 1: Classify as DR vs No_DR
     - Stage 2: Classify DR severity (Mild/Moderate/Severe/Proliferative)

3. **Skip Phase 3 (Label Smoothing)**
   - Given CBAM's failure, label smoothing is unlikely to help
   - It may further degrade the well-separated feature space

---

## Conclusion

**CBAM integration was unsuccessful** for this specific task. The attention mechanism:
- ❌ Failed to improve fine-grained classification (Mild class collapsed)
- ❌ Degraded OSR performance (AUROC dropped below 99%)
- ❌ Did not meet the project goals

**Recommendation**: **Revert to baseline** and explore alternative approaches (Grad-CAM, ensemble, data-centric) if Mild/Severe improvement is still required. The baseline model (86.23% accuracy, 99.42% AUROC) is a strong, production-ready solution.

---

## Artifacts
- **CBAM Model**: `checkpoints/resnet50_full_5class.pth`
- **Training Report**: `focal_loss_training_report.txt`
- **OSR Evaluation**: `cbam_osr_results.txt`
- **Confusion Matrix**: `focal_loss_confusion_matrix.png`
