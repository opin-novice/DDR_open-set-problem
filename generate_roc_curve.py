import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset

def extract_features(model, loader, device):
    """Extract penultimate layer features"""
    features = []
    labels = []
    
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            # Extract features (penultimate layer)
            x = model.conv1(data)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            f = torch.flatten(x, 1)
            features.append(f.cpu().numpy())
            labels.append(target.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def compute_mahalanobis_scores(features, class_means, precision):
    """Compute minimum Mahalanobis distance for each sample"""
    scores = []
    for feat in features:
        dists = []
        for c in range(len(class_means)):
            centered = feat - class_means[c]
            d = np.sqrt(np.sum(centered @ precision * centered))
            dists.append(d)
        scores.append(min(dists))
    return np.array(scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = 'outputs/roc_curves'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Model
    print("Loading model...")
    model_path = 'checkpoints/resnet50_full_5class.pth'
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load Datasets
    print("Loading datasets...")
    ddr_train = DDR(root='DDR dataset', train=True, transform=transform, 
                    train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_test = DDR(root='DDR dataset', train=False, transform=transform, 
                   train_class_num=5, test_class_num=5, includes_all_train_class=True)
    acrima = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
    
    train_loader = DataLoader(ddr_train, batch_size=32, shuffle=False, num_workers=4)
    test_known_loader = DataLoader(ddr_test, batch_size=32, shuffle=False, num_workers=4)
    test_unknown_loader = DataLoader(acrima, batch_size=32, shuffle=False, num_workers=4)
    
    # Extract Features
    print("Extracting features from training set...")
    train_features, train_labels = extract_features(model, train_loader, device)
    
    print("Extracting features from test sets...")
    known_features, _ = extract_features(model, test_known_loader, device)
    unknown_features, _ = extract_features(model, test_unknown_loader, device)
    
    # Compute Mahalanobis Statistics
    print("Computing Mahalanobis statistics...")
    class_means = []
    centered_data = []
    for c in range(5):
        mask = train_labels == c
        c_feats = train_features[mask]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_data.append(c_feats - mean)
        
    cov = EmpiricalCovariance().fit(np.concatenate(centered_data))
    precision = cov.precision_
    
    # Compute Scores
    print("Computing Mahalanobis scores...")
    known_scores = compute_mahalanobis_scores(known_features, class_means, precision)
    unknown_scores = compute_mahalanobis_scores(unknown_features, class_means, precision)
    
    # Prepare labels: 0 = Known (Inlier), 1 = Unknown (Outlier)
    y_true = np.concatenate([
        np.zeros(len(known_scores)),  # Known samples
        np.ones(len(unknown_scores))   # Unknown samples
    ])
    
    # Scores (higher distance = more likely to be unknown)
    y_scores = np.concatenate([known_scores, unknown_scores])
    
    # Compute ROC curve
    print("Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC Curve
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Mark optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, 
             label=f'Optimal Threshold = {optimal_threshold:.2f}\n(TPR={optimal_tpr:.3f}, FPR={optimal_fpr:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('ROC Curve: Open Set Recognition\n(Known: DDR vs Unknown: Glaucoma)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nROC curve saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("ROC CURVE STATISTICS")
    print("="*60)
    print(f"AUROC: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"  - True Positive Rate (Sensitivity): {optimal_tpr:.4f} ({optimal_tpr*100:.2f}%)")
    print(f"  - False Positive Rate: {optimal_fpr:.4f} ({optimal_fpr*100:.2f}%)")
    print(f"  - Specificity: {1-optimal_fpr:.4f} ({(1-optimal_fpr)*100:.2f}%)")
    print("="*60)
    
    # Save numerical results
    results_path = os.path.join(output_dir, 'roc_statistics.txt')
    with open(results_path, 'w') as f:
        f.write("ROC Curve Statistics\n")
        f.write("="*60 + "\n")
        f.write(f"AUROC: {roc_auc:.4f} ({roc_auc*100:.2f}%)\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.2f}\n")
        f.write(f"  - True Positive Rate (Sensitivity): {optimal_tpr:.4f} ({optimal_tpr*100:.2f}%)\n")
        f.write(f"  - False Positive Rate: {optimal_fpr:.4f} ({optimal_fpr*100:.2f}%)\n")
        f.write(f"  - Specificity: {1-optimal_fpr:.4f} ({(1-optimal_fpr)*100:.2f}%)\n")
        f.write("="*60 + "\n")
        f.write(f"\nKnown samples: {len(known_scores)}\n")
        f.write(f"Unknown samples: {len(unknown_scores)}\n")
    
    print(f"Statistics saved to: {results_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()
