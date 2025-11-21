"""
Mahalanobis Distance-Based OOD Detection
Follows the high-level approach for feature-space OOD detection
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Get features before final FC layer
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        
        if return_features:
            return features
        
        logits = self.backbone.fc(features)
        return logits

def extract_features(model, dataloader, use_gpu=True):
    """Extract penultimate layer features"""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu:
                data = data.cuda()
            
            features = model(data, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_features), np.concatenate(all_labels)

def compute_class_statistics(features, labels, num_classes=3, epsilon=0.01):
    """
    Compute class means and pooled covariance
    
    Returns:
        class_means: list of mean vectors for each class
        precision: inverse of pooled covariance matrix (with regularization)
    """
    class_means = []
    all_centered_features = []
    
    # Compute class means
    for c in range(num_classes):
        class_mask = labels == c
        class_features = features[class_mask]
        class_mean = class_features.mean(axis=0)
        class_means.append(class_mean)
        
        # Center features for covariance computation
        centered = class_features - class_mean
        all_centered_features.append(centered)
        
        print(f"  Class {c}: {class_mask.sum()} samples")
    
    # Compute pooled covariance
    all_centered = np.concatenate(all_centered_features, axis=0)
    
    # Use empirical covariance with regularization
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(all_centered)
    covariance = cov_estimator.covariance_
    
    # Add regularization: Σ + ε*I
    feature_dim = covariance.shape[0]
    covariance_reg = covariance + epsilon * np.eye(feature_dim)
    
    # Compute precision matrix (inverse)
    precision = np.linalg.inv(covariance_reg)
    
    return class_means, precision

def mahalanobis_distance(features, class_mean, precision):
    """
    Compute Mahalanobis distance: sqrt((x - μ)^T Σ^{-1} (x - μ))
    
    Args:
        features: (N, D) array of features
        class_mean: (D,) class mean vector
        precision: (D, D) inverse covariance matrix
    
    Returns:
        distances: (N,) array of Mahalanobis distances
    """
    # Center features
    centered = features - class_mean
    
    # Compute: (x - μ)^T Σ^{-1} (x - μ)
    # This is vectorized for all samples at once
    mahal_sq = np.sum(centered @ precision * centered, axis=1)
    
    return np.sqrt(np.abs(mahal_sq))  # abs for numerical stability

def compute_ood_scores(features, class_means, precision):
    """
    Compute OOD score = minimum Mahalanobis distance to any class
    
    Lower distance = more likely to be known
    Higher distance = more likely to be unknown (OOD)
    """
    num_samples = features.shape[0]
    num_classes = len(class_means)
    
    # Compute distance to each class
    distances = np.zeros((num_samples, num_classes))
    
    for c in range(num_classes):
        distances[:, c] = mahalanobis_distance(features, class_means[c], precision)
    
    # OOD score = minimum distance across all classes
    # Higher min distance = further from all known classes = more likely OOD
    ood_scores = distances.min(axis=1)
    
    return ood_scores, distances

def main():
    print("="*80)
    print("MAHALANOBIS DISTANCE OOD DETECTION")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # Load model
    model = ResNet50Classifier(num_classes=3)
    
    if os.path.exists('checkpoints/focal_closed_set.pth'):
        model.load_state_dict(torch.load('checkpoints/focal_closed_set.pth'))
        print("\nLoaded: focal_closed_set.pth")
    elif os.path.exists('checkpoints/closed_set_model.pth'):
        model.load_state_dict(torch.load('checkpoints/closed_set_model.pth'))
        print("\nLoaded: closed_set_model.pth")
    else:
        print("ERROR: No trained model found!")
        return
    
    if use_gpu:
        model = model.cuda()
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load datasets
    trainset = DDR(root='DDR dataset', train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    testset = DDR(root='DDR dataset', train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    # Step 1: Extract features from training set
    print("\nStep 1: Extracting training features...")
    train_features, train_labels = extract_features(model, trainloader, use_gpu)
    
    # Filter to known classes only
    known_train_mask = train_labels < 3
    train_features_known = train_features[known_train_mask]
    train_labels_known = train_labels[known_train_mask]
    
    print(f"  Total training features: {len(train_features_known)}")
    print(f"  Feature dimension: {train_features_known.shape[1]}")
    
    # Step 2: Compute class statistics
    print("\nStep 2: Computing class means and pooled covariance...")
    class_means, precision = compute_class_statistics(
        train_features_known, 
        train_labels_known, 
        num_classes=3,
        epsilon=0.01  # Regularization
    )
    
    print(f"  Precision matrix shape: {precision.shape}")
    print(f"  Condition number: {np.linalg.cond(precision):.2e}")
    
    # Step 3: Extract test features
    print("\nStep 3: Extracting test features...")
    test_features, test_labels = extract_features(model, testloader, use_gpu)
    
    known_mask = test_labels < 3
    unknown_mask = test_labels >= 3
    
    print(f"  Known samples: {known_mask.sum()}")
    print(f"  Unknown samples: {unknown_mask.sum()}")
    
    # Step 4: Compute Mahalanobis distances
    print("\nStep 4: Computing Mahalanobis distances...")
    ood_scores, all_distances = compute_ood_scores(test_features, class_means, precision)
    
    # Step 5: Evaluate known class accuracy (using distances)
    print("\n" + "="*80)
    print("CLOSED-SET CLASSIFICATION (Mahalanobis-based)")
    print("="*80)
    
    # Predict class with minimum distance
    predictions = all_distances.argmin(axis=1)
    
    known_preds = predictions[known_mask]
    known_true = test_labels[known_mask]
    known_acc = 100.0 * (known_preds == known_true).sum() / len(known_true)
    
    print(f"\nOverall Known Accuracy: {known_acc:.2f}%")
    
    for c in range(3):
        class_mask = test_labels == c
        if class_mask.sum() > 0:
            class_preds = predictions[class_mask]
            class_acc = 100.0 * (class_preds == c).sum() / class_mask.sum()
            print(f"  Class {c}: {class_acc:.2f}%")
    
    # Step 6: OOD Detection
    print("\n" + "="*80)
    print("OOD DETECTION (Mahalanobis)")
    print("="*80)
    
    # Binary labels: 0 = known, 1 = unknown
    binary_labels = unknown_mask.astype(int)
    
    # Compute AUROC (higher score = more likely OOD)
    auroc = roc_auc_score(binary_labels, ood_scores) * 100
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(binary_labels, ood_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nAUROC: {auroc:.2f}%")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"  TPR: {tpr[optimal_idx]:.2%}")
    print(f"  FPR: {fpr[optimal_idx]:.2%}")
    
    # Distance statistics
    print(f"\nMahalanobis Distance Statistics:")
    print(f"  Known   - Mean: {ood_scores[known_mask].mean():.4f}, Std: {ood_scores[known_mask].std():.4f}")
    print(f"  Unknown - Mean: {ood_scores[unknown_mask].mean():.4f}, Std: {ood_scores[unknown_mask].std():.4f}")
    
    # Combined metric
    combined = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Known Class Accuracy: {known_acc:.2f}%")
    print(f"Unknown Detection AUROC: {auroc:.2f}%")
    print(f"Combined Score (H-Mean): {combined:.2f}%")
    print("="*80)
    
    # Success check
    if known_acc >= 85 and auroc >= 80:
        print("\n SUCCESS! Both metrics meet targets!")
    elif auroc >= 80:
        print("\n GOOD! AUROC meets target. Known accuracy needs improvement.")
    elif known_acc >= 85:
        print("\n PARTIAL: Known accuracy good, but AUROC needs improvement.")
    else:
        print("\n Both metrics need improvement.")
    
    return known_acc, auroc, combined

if __name__ == '__main__':
    main()
