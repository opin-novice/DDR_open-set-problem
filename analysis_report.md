# Analysis of Mahalanobis + OE Results

## Current Performance
*   **AUROC:** 86.59% (✅ **Excellent** - Meets target >85%)
*   **Known Accuracy:** 78.82% (❌ **Too Low** - Target >85-90%)

## Diagnosis
The high AUROC confirms that **Mahalanobis + Outlier Exposure** is the correct strategy for detecting unknowns. However, the low accuracy suggests:
1.  **Over-Regularization:** The Outlier Exposure loss weight (`lambda_oe = 0.5`) might be too high, forcing the model to be too uncertain even on hard known samples (which look like outliers).
2.  **LogitNorm Temperature:** The temperature (`temp = 0.1`) might be too low, squashing the logits too much and making optimization difficult for the cross-entropy loss.
3.  **Mixup Aggressiveness:** Using Mixup as OE might be confusing the model if the mixed images look too much like valid "Mild/Moderate" transitions.

## Proposed Adjustments (Retraining Plan)

We need to **relax the regularization** to recover Known Accuracy while maintaining the high AUROC.

### 1. Reduce OE Weight
*   **Current:** `lambda_oe = 0.5`
*   **New:** `lambda_oe = 0.1`
*   **Reason:** Allow the model to prioritize Cross-Entropy (accuracy) more.

### 2. Increase Temperature
*   **Current:** `temp = 0.1`
*   **New:** `temp = 1.0` (or remove LogitNorm and use standard Softmax)
*   **Reason:** `temp=0.1` is very aggressive. Standard LogitNorm papers often use learnable alpha or higher temp. Let's try `0.5` or `1.0`.

### 3. Refine OE Proxy
*   **Current:** Mixup + Noise
*   **New:** **Noise Only** (or Jigsaw)
*   **Reason:** Mixup between Class 0 (No DR) and Class 1 (Mild DR) creates a valid "Very Mild DR" image. Forcing the model to output "Unknown" on this valid transition might be hurting accuracy. Random noise is safer.

## Action Plan
I will modify `train_oe_mahalanobis.py` to:
1.  Set `lambda_oe = 0.1`
2.  Set `temp = 0.5`
3.  Use **only Noise** as OE (disable Mixup OE for now to be safe).
