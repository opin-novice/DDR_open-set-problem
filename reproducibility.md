# Cross-Dataset OSR Reproducibility Report

## Experiment Setup
### Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Epochs** | 30 |
| **Batch Size** | 32 |
| **Optimizer** | Adam (lr=0.001) |
| **Scheduler** | Cosine Annealing (T_max=30) |
| **Augmentation** | RandomFlip, RandomRotation(10), Resize(224) |
| **Backbone** | ResNet50 (ImageNet Pretrained) |
| **Seeds** | [42, 1, 2024] |

### Dataset Split
- **Known Classes:** 5 (No_DR, Mild, Moderate, Severe, Proliferative)
- **Unknown Class:** Glaucoma (ACRIMA Dataset)

## Aggregated Results (Mean ± Std)
| Metric | Mean | Std |
| :--- | :--- | :--- |
| **Known Accuracy** | **94.43%** | ±0.36 |
| **AUROC** | **99.61%** | ±0.19 |
| **H-mean** | **96.95%** | ±0.14 |

## Visualizations
### 1. Class-wise Accuracy Stability
![Class-wise Accuracy Boxplot](reproducibility_class_accuracy_boxplot.png)
*Boxplot showing the spread of accuracy for each class across 3 random seeds. Tight boxes indicate high stability.*

### 2. OOD Score Separation
![Distance Histogram](reproducibility_distance_histogram.png)
*Histogram of Mahalanobis distances aggregated across all 3 seeds. Clear separation between Blue (Known) and Red (Unknown) indicates robust OOD detection.*

## Detailed Results Table
|    Seed |   Known Acc |   AUROC |   H-mean |   No_DR |   Mild |   Moderate |   Severe |   Proliferative |
|--------:|------------:|--------:|---------:|--------:|-------:|-----------:|---------:|----------------:|
|   42.00 |       94.55 |   99.41 |    96.92 |   96.60 |  90.16 |      92.43 |    88.14 |           95.62 |
|    1.00 |       94.02 |   99.79 |    96.82 |   96.11 |  90.48 |      91.80 |    86.44 |           94.96 |
| 2024.00 |       94.71 |   99.62 |    97.10 |   96.90 |  91.75 |      92.14 |    90.25 |           95.51 |

## Class-wise Accuracy (Known)
| Class | Mean Acc | Std |
| :--- | :--- | :--- |
| No_DR | 96.54% | ±0.40 |
| Mild | 90.79% | ±0.84 |
| Moderate | 92.12% | ±0.31 |
| Severe | 88.28% | ±1.91 |
| Proliferative | 95.36% | ±0.35 |
