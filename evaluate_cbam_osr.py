import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.covariance import EmpiricalCovariance
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset
from models.resnet_cbam import resnet50_cbam

def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            
            # Extract penultimate features
            x = model.conv1(inputs)
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
    print("EVALUATING CBAM MODEL ON OSR TASK")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load CBAM Model
    model_path = 'checkpoints/resnet50_full_5class.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading CBAM model from {model_path}...")
    model = resnet50_cbam(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    # 2. Setup DataLoaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Training set (for Mahalanobis stats)
    trainset = DDR(root='DDR dataset', split='train', transform=transform,
                   train_class_num=5, test_class_num=5, includes_all_train_class=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)
    
    # Test set (Known)
    testset = DDR(root='DDR dataset', split='test', transform=transform,
                  train_class_num=5, test_class_num=5, includes_all_train_class=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    # Unknown set (ACRIMA Glaucoma)
    unknownset = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
    unknownloader = DataLoader(unknownset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Extract Features
    print("\nExtracting training features...")
    train_feats, train_labels = extract_features(model, trainloader, device)
    
    print("Extracting test features (Known)...")
    test_feats_known, test_labels_known = extract_features(model, testloader, device)
    
    print("Extracting unknown features (ACRIMA)...")
    test_feats_unknown, _ = extract_features(model, unknownloader, device)
    
    # 4. Compute Mahalanobis Statistics
    print("\nComputing Mahalanobis statistics...")
    class_means, precision = compute_mahalanobis_stats(train_feats, train_labels, num_classes=5)
    
    # 5. Compute OOD Scores
    print("Computing OOD scores...")
    scores_known = mahalanobis_score(test_feats_known, class_means, precision)
    scores_unknown = mahalanobis_score(test_feats_unknown, class_means, precision)
    
    # 6. Evaluate AUROC
    y_true = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
    y_scores = np.concatenate([scores_known, scores_unknown])
    auroc = roc_auc_score(y_true, y_scores) * 100
    
    # 7. Closed-Set Accuracy
    model.eval()
    correct = 0
    total = 0
    class_correct = np.zeros(5)
    class_total = np.zeros(5)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            for i in range(5):
                mask = labels == i
                if mask.sum() > 0:
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
    
    acc = 100.0 * correct / total
    class_acc = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(5)]
    
    # 8. Print Results
    print("\n" + "="*80)
    print("CBAM MODEL - OSR EVALUATION RESULTS")
    print("="*80)
    print(f"\nClosed-Set Accuracy: {acc:.2f}%")
    print("\nPer-Class Accuracy:")
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    for i, name in enumerate(class_names):
        print(f"  {name:15s}: {class_acc[i]:5.2f}%")
    
    print(f"\nOSR AUROC: {auroc:.2f}%")
    print("="*80)
    
    # Save results
    with open('cbam_osr_results.txt', 'w') as f:
        f.write("CBAM Model - OSR Evaluation Results\n")
        f.write("="*60 + "\n")
        f.write(f"Closed-Set Accuracy: {acc:.2f}%\n")
        f.write("\nPer-Class Accuracy:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name:15s}: {class_acc[i]:5.2f}%\n")
        f.write(f"\nOSR AUROC: {auroc:.2f}%\n")
    
    print("\nResults saved to: cbam_osr_results.txt")

if __name__ == "__main__":
    main()
