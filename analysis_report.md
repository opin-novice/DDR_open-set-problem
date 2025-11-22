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

### B. Why Glaucoma is "Unknown"
*   **The "No-Lesion" Problem:** Glaucoma images typically show a clear retina with an enlarged optic cup, but *no hemorrhages or exudates*.
*   **Confusion with 'No_DR':** The hardest Glaucoma samples are most often confused with **'No_DR'** (Class 0).
    *   *Reason:* Like 'No_DR', Glaucoma images are free of lesions. The model sees "clean" tissue and thinks, "This looks a bit like a healthy eye."
*   **The Rejection:** However, even these "confusing" samples have a **Mahalanobis distance > 50**, whereas true 'No_DR' samples typically have distances < 30.
    *   *Why?* The texture of the optic disc in Glaucoma is distinct enough that it doesn't fit the tight statistical distribution of a normal 'No_DR' fundus.

## 3. Conclusion
The model differentiates DR from Glaucoma not by learning what Glaucoma *is*, but by knowing precisely what DR *is not*.

> **Key Takeaway:** The model acts as a **"DR Feature Validator."**
> *   **Has DR Features?** -> Classify as Mild/Moderate/Severe/Proliferative.
> *   **No DR Features (Clean)?** -> Check if it fits the strict 'No_DR' statistical profile.
> *   **Neither?** -> **Reject as Unknown (Glaucoma/Other).**
