import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            # Hook to get penultimate features
            # We need to redefine the forward pass or use a hook
            # Since we just replaced fc, we can use the backbone parts
            
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
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
    return np.concatenate(all_features), np.concatenate(all_labels)

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
    print("CROSS-DATASET OSR EVALUATION (DDR vs ACRIMA)")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model_path = 'checkpoints/resnet50_full_5class.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run training first.")
        return

    print(f"Loading model from {model_path}...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 2. Data Loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Known: DDR Test Set (All 5 classes)
    ddr_set = DDR(root='DDR dataset', train=False, transform=transform, 
                  train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_loader = DataLoader(ddr_set, batch_size=32, shuffle=False, num_workers=4)
    
    # Unknown: ACRIMA (Glaucoma only)
    acrima_set = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
    acrima_loader = DataLoader(acrima_set, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. Extract Features
    print("\nExtracting features from DDR (Known)...")
    # We need training features to compute class statistics
    ddr_train_set = DDR(root='DDR dataset', train=True, transform=transform, 
                        train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_train_loader = DataLoader(ddr_train_set, batch_size=32, shuffle=False, num_workers=4)
    
    train_feats, train_labels = extract_features(model, ddr_train_loader, device)
    test_feats_known, test_labels_known = extract_features(model, ddr_loader, device)
    
    print("\nExtracting features from ACRIMA (Unknown)...")
    test_feats_unknown, _ = extract_features(model, acrima_loader, device)
    
    # 4. Compute Statistics
    print("\nComputing Mahalanobis statistics...")
    class_means, precision = compute_mahalanobis_stats(train_feats, train_labels, num_classes=5)
    
    # 5. Compute Scores
    print("Computing OOD scores...")
    scores_known = mahalanobis_score(test_feats_known, class_means, precision)
    scores_unknown = mahalanobis_score(test_feats_unknown, class_means, precision)
    
    # --- NEW: Inspect Hardest Samples ---
    # Find Glaucoma samples with LOWEST Mahalanobis distance (most similar to DR)
    # We need the filenames for this. Let's modify extract_features or just reload the dataset to get paths.
    # Since we can't easily change extract_features return signature without breaking things, 
    # let's just iterate the dataset again for the top 10 indices.
    
    print("\nInspecting Hardest Glaucoma Samples...")
    hardest_indices = np.argsort(scores_unknown)[:10] # Lowest distances
    
    with open('hardest_glaucoma_samples.txt', 'w') as f:
        f.write("Top 10 'Hardest' Glaucoma Samples (Most similar to DR)\n")
        f.write("======================================================\n")
        
        for rank, idx in enumerate(hardest_indices):
            # Get the image path from the dataset
            img_path = acrima_set.images[idx]
            score = scores_unknown[idx]
            
            # Find which DR class it was closest to
            feat = test_feats_unknown[idx]
            dists = []
            for c in range(5):
                centered = feat - class_means[c]
                d = np.sqrt(np.sum(centered @ precision * centered))
                dists.append(d)
            closest_class = np.argmin(dists)
            class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
            
            print(f"Rank {rank+1}: {os.path.basename(img_path)} | Score: {score:.2f} | Confused with: {class_names[closest_class]}")
            f.write(f"Rank {rank+1}: {os.path.basename(img_path)}\n")
            f.write(f"  - Mahalanobis Distance: {score:.4f}\n")
            f.write(f"  - Closest DR Class: {class_names[closest_class]}\n")
            f.write(f"  - Path: {img_path}\n\n")

    # 6. Evaluate
    y_true = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
    y_scores = np.concatenate([scores_known, scores_unknown])
    
    auroc = roc_auc_score(y_true, y_scores) * 100
    
    # --- NEW METRICS ---
    
    # 1. Known Accuracy (Closed Set)
    # We need to predict class for known samples using Mahalanobis distance
    # The class with min distance is the predicted class
    num_samples = test_feats_known.shape[0]
    num_classes = 5
    distances = np.zeros((num_samples, num_classes))
    for c in range(num_classes):
        centered = test_feats_known - class_means[c]
        mahal_sq = np.sum(centered @ precision * centered, axis=1)
        distances[:, c] = np.sqrt(mahal_sq)
    
    predictions = distances.argmin(axis=1)
    known_acc = 100.0 * (predictions == test_labels_known).sum() / len(test_labels_known)
    
    # 2. H-Mean (Combined Score)
    h_mean = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
    
    # 3. Confusion Matrix (Known Classes)
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(test_labels_known, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'],
                yticklabels=['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Known Accuracy: {known_acc:.2f}%)')
    plt.savefig('cross_dataset_confusion_matrix.png')
    plt.close()
    
    # 4. Distance Distribution Overlap
    plt.figure(figsize=(10, 6))
    plt.hist(scores_known, bins=50, alpha=0.5, label='Known (DDR)', color='blue', density=True)
    plt.hist(scores_unknown, bins=50, alpha=0.5, label='Unknown (Glaucoma)', color='red', density=True)
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Density')
    plt.title(f'OOD Detection (AUROC: {auroc:.2f}%)')
    plt.legend()
    plt.savefig('cross_dataset_distribution.png')
    plt.close()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Known Samples (DDR): {len(scores_known)}")
    print(f"Unknown Samples (ACRIMA): {len(scores_unknown)}")
    print(f"Known Accuracy: {known_acc:.2f}%")
    print(f"AUROC: {auroc:.2f}%")
    print(f"H-mean: {h_mean:.2f}%")
    print("="*80)
    
    # Save results
    with open('cross_dataset_results.txt', 'w') as f:
        f.write(f"Cross-Dataset OSR Results\n")
        f.write(f"Known: DDR (5 classes)\n")
        f.write(f"Unknown: ACRIMA (Glaucoma)\n")
        f.write(f"Known Accuracy: {known_acc:.2f}%\n")
        f.write(f"AUROC: {auroc:.2f}%\n")
        f.write(f"H-mean: {h_mean:.2f}%\n")

if __name__ == "__main__":
    main()
