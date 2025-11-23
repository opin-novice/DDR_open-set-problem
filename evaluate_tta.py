import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.covariance import EmpiricalCovariance
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset

def get_tta_transforms():
    """
    Define 5-view TTA transforms:
    1. Original
    2. Horizontal Flip
    3. Rotate +5
    4. Rotate -5
    5. Horizontal Flip + Rotate -5
    """
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transforms_list = [
        # 1. Original
        base_transform,
        
        # 2. Horizontal Flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        
        # 3. Rotate +5
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((5, 5)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        
        # 4. Rotate -5
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        
        # 5. HFlip + Rotate -5
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    ]
    return transforms_list

def predict_tta(model, image, transforms_list, device):
    """
    Apply TTA to a single image (or batch of images)
    Returns averaged logits and averaged features
    """
    # This function assumes 'image' is the raw PIL image from dataset
    # But DataLoader returns tensors. 
    # We need to apply TTA at the dataset level or manually apply transforms to the original image.
    # Since standard DataLoader applies one transform, we need a custom approach.
    pass

# Better approach: Custom Dataset Wrapper or just iterate and apply transforms manually if we have access to PIL images.
# DDR dataset returns PIL images if transform is None? No, it applies transform.
# Let's modify the flow:
# We will create 5 DataLoaders, one for each transform.
# We iterate them in lockstep and average the results.

def extract_features_tta(model, dataloaders, device):
    model.eval()
    all_features = []
    all_labels = []
    all_probs = []
    
    print(f"Running TTA with {len(dataloaders)} views...")
    
    with torch.no_grad():
        # Zip all dataloaders together
        for batches in zip(*dataloaders):
            # batches is a tuple of (data, label) tuples
            # Check labels consistency
            labels = batches[0][1]
            
            # Collect outputs for this batch across all views
            batch_features_sum = None
            batch_probs_sum = None
            
            for i, (data, _) in enumerate(batches):
                data = data.to(device)
                
                # Forward pass for features
                x = model.conv1(data)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                features = torch.flatten(x, 1)
                
                # Forward pass for logits/probs
                logits = model.fc(features)
                probs = F.softmax(logits, dim=1)
                
                if batch_features_sum is None:
                    batch_features_sum = features
                    batch_probs_sum = probs
                else:
                    batch_features_sum += features
                    batch_probs_sum += probs
            
            # Average
            avg_features = batch_features_sum / len(dataloaders)
            avg_probs = batch_probs_sum / len(dataloaders)
            
            all_features.append(avg_features.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(avg_probs.cpu().numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels), np.concatenate(all_probs)

def compute_mahalanobis_stats(features, labels, num_classes=5):
    class_means = []
    all_centered_features = []
    
    for c in range(num_classes):
        class_mask = labels == c
        class_features = features[class_mask]
        class_mean = class_features.mean(axis=0)
        class_means.append(class_mean)
        
        centered = class_features - class_mean
        all_centered_features.append(centered)
        
    all_centered = np.concatenate(all_centered_features, axis=0)
    
    cov_estimator = EmpiricalCovariance()
    cov_estimator.fit(all_centered)
    precision = cov_estimator.precision_
    
    return class_means, precision

def mahalanobis_score(features, class_means, precision):
    num_samples = features.shape[0]
    num_classes = len(class_means)
    distances = np.zeros((num_samples, num_classes))
    
    for c in range(num_classes):
        centered = features - class_means[c]
        mahal_sq = np.sum(centered @ precision * centered, axis=1)
        distances[:, c] = np.sqrt(mahal_sq)
        
    return distances.min(axis=1)

def main():
    print("="*80)
    print("EVALUATING TEST TIME AUGMENTATION (TTA)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model_path = 'checkpoints/resnet50_full_5class.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 2. Setup TTA DataLoaders
    transforms_list = get_tta_transforms()
    
    # DDR Test Loaders (Known)
    ddr_loaders = []
    for t in transforms_list:
        ds = DDR(root='DDR dataset', split='test', transform=t, 
                 train_class_num=5, test_class_num=5, includes_all_train_class=True)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        ddr_loaders.append(dl)
        
    # ACRIMA Loaders (Unknown)
    acrima_loaders = []
    for t in transforms_list:
        ds = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=t)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
        acrima_loaders.append(dl)
        
    # DDR Train Loaders (for Mahalanobis stats) - No TTA needed here, just standard transform
    # Actually, should we use TTA for training stats? Usually no, just standard training distribution.
    # Let's use standard transform for training stats to be consistent with "prototypes".
    base_transform = transforms_list[0]
    ddr_train_set = DDR(root='DDR dataset', split='train', transform=base_transform, 
                        train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_train_loader = DataLoader(ddr_train_set, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Extract Features & Probs
    print("\n--- Phase 1: Closed Set Evaluation (DDR) ---")
    test_feats_known, test_labels_known, test_probs_known = extract_features_tta(model, ddr_loaders, device)
    
    # Calculate Accuracy
    preds = np.argmax(test_probs_known, axis=1)
    acc = 100.0 * (preds == test_labels_known).sum() / len(test_labels_known)
    print(f"\nTTA Accuracy (DDR): {acc:.2f}%")
    
    # Debug: Check label distribution
    unique, counts = np.unique(test_labels_known, return_counts=True)
    print("\nLabel Distribution in Evaluation:")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c}")

    # Per-class Accuracy
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print("\nPer-Class Accuracy with TTA:")
    # Explicitly provide labels to ensure 5x5 matrix even if some classes are missing
    cm = confusion_matrix(test_labels_known, preds, labels=range(5))
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total > 0:
            class_acc = 100.0 * cm[i, i] / total
            print(f"  {name:15s}: {class_acc:5.2f}% ({cm[i, i]}/{total})")
        else:
            print(f"  {name:15s}:   N/A% (0/0)")
        
    # 4. OSR Evaluation
    print("\n--- Phase 2: Open Set Evaluation (DDR vs ACRIMA) ---")
    
    # Get Training Stats (No TTA for prototypes)
    print("Extracting training features for Mahalanobis stats...")
    # We can reuse extract_features_tta with a single loader list
    train_feats, train_labels, _ = extract_features_tta(model, [ddr_train_loader], device)
    
    print("Computing Mahalanobis statistics...")
    class_means, precision = compute_mahalanobis_stats(train_feats, train_labels, num_classes=5)
    
    # Extract Unknown Features (with TTA)
    print("Extracting ACRIMA features with TTA...")
    test_feats_unknown, _, _ = extract_features_tta(model, acrima_loaders, device)
    
    # Compute Scores
    print("Computing OOD scores...")
    scores_known = mahalanobis_score(test_feats_known, class_means, precision)
    scores_unknown = mahalanobis_score(test_feats_unknown, class_means, precision)
    
    # Evaluate AUROC
    y_true = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
    y_scores = np.concatenate([scores_known, scores_unknown])
    auroc = roc_auc_score(y_true, y_scores) * 100
    
    print(f"\nTTA AUROC: {auroc:.2f}%")
    
    # Save results
    with open('tta_results.txt', 'w') as f:
        f.write(f"TTA Results (5-view)\n")
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(f"AUROC: {auroc:.2f}%\n")
        f.write("Per-Class Accuracy:\n")
        for i, name in enumerate(class_names):
            class_acc = 100.0 * cm[i, i] / cm[i].sum()
            f.write(f"  {name}: {class_acc:.2f}%\n")

if __name__ == "__main__":
    main()
