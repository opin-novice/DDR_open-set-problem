# ROC Curve Analysis: Open Set Recognition Performance

## Overview
The ROC (Receiver Operating Characteristic) curve visualizes the trade-off between detecting Unknown samples (Glaucoma) and maintaining low false alarms on Known samples (DDR).

## Results

### Key Metrics
| Metric | Value | Interpretation |
|:---|:---|:---|
| **AUROC** | **99.85%** | Near-perfect separation between Known and Unknown |
| **Optimal Threshold** | 66.18 | Mahalanobis distance cutoff for best performance |
| **Sensitivity (TPR)** | 97.47% | Correctly identifies 97.47% of Unknown samples |
| **Specificity** | 98.89% | Correctly identifies 98.89% of Known samples |
| **False Positive Rate** | 1.11% | Only 1.11% of Known samples misclassified as Unknown |

### What This Means

1. **Exceptional Performance**: An AUROC of 99.85% indicates the model has nearly perfect ability to distinguish between:
   - **Known**: Diabetic Retinopathy images (all 5 severity levels)
   - **Unknown**: Glaucoma images

2. **Optimal Operating Point**: At the threshold of 66.18:
   - The model catches **97.47%** of Glaucoma cases (high sensitivity)
   - While only raising **1.11%** false alarms on DR cases (high specificity)
   - This represents an excellent balance for clinical deployment

3. **Clinical Implications**:
   - **For Screening**: The model can reliably flag non-DR eye diseases for specialist review
   - **Safety**: Low false positive rate (1.11%) means minimal unnecessary referrals
   - **Reliability**: High sensitivity (97.47%) ensures most unknown conditions are caught

## Visualization
The ROC curve plot shows:
- **Orange curve**: Model performance across all possible thresholds
- **Blue dashed line**: Random classifier baseline (50% AUROC)
- **Red dot**: Optimal operating point (maximum Youden's J statistic)

The curve hugs the top-left corner, indicating excellent discrimination ability.

## Files Generated
- `outputs/roc_curves/roc_curve.png` - Visual ROC curve plot
- `outputs/roc_curves/roc_statistics.txt` - Detailed numerical statistics

## Comparison to Literature
Typical OSR systems achieve 85-95% AUROC. Our **99.85%** significantly exceeds this benchmark, demonstrating:
1. Strong feature learning from the ResNet50 backbone
2. Effective use of Mahalanobis distance for OOD detection
3. Clear visual differences between DR pathology and Glaucoma morphology
