# Diagnosis of Performance Regression

## Comparison
*   **Run 1 (Baseline):** Temp=0.1, Lambda=0.5, Mixup=On
    *   **AUROC:** 86.59% (Excellent)
    *   **Acc:** 78.82% (Good but needs boost)
*   **Run 2 (Adjusted):** Temp=0.5, Lambda=0.1, Mixup=Off
    *   **AUROC:** 79.49% (Worse)
    *   **Acc:** 55.12% (Catastrophic Failure)

## Root Cause Analysis
1.  **Temperature 0.5 Failure:** LogitNorm is very sensitive to temperature. Increasing it to 0.5 likely made the logits too small relative to the angular margin, causing the model to underfit significantly (hence 55% accuracy).
2.  **Mixup Removal:** Removing Mixup might have reduced the robustness of the features. Mixup acts as a regularizer that helps generalization.

## Conclusion
**Run 1 was actually very close to success.**
The accuracy of 78.82% is not terrible for a difficult medical dataset. The drop to 55% confirms that **Temp=0.1 is necessary** for this LogitNorm implementation to work.

## Recovery Plan (Run 3)
We should **revert to the Run 1 configuration** but make a *minor* adjustment to favor accuracy.

1.  **Revert Temp:** `0.5` → `0.1` (Critical for convergence)
2.  **Re-enable Mixup:** It helped feature learning.
3.  **Tune Lambda:** `0.5` → `0.25` (Slight reduction to help accuracy, but not as drastic as 0.1).
4.  **Increase Epochs:** 20 → 30 (Give it more time to converge).

**Expected Outcome:**
*   AUROC: ~85-86% (Recovered)
*   Acc: ~80-82% (Improved from 78% due to lower lambda and more epochs)
