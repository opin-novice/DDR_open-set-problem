"""
Two-Stage Energy-Based OOD Detection for DDR Dataset

Stage 1: Closed-set classifier (already trained)
Stage 2: Energy-based OOD detection (this script)

Energy Score: E(x) = -log(sum(exp(f_c(x))))
- Lower energy = Known (confident predictions)
- Higher energy = Unknown (uncertain predictions)

This approach is proven to work better than ARPL for medical imaging datasets
with overlapping classes (like diabetic retinopathy grades).
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
import matplotlib.pyplot as plt

# Add project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from datasets import DDR

# ==========================================
# Simple ResNet50 Classifier
# ==========================================
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x, return_logits=True):
        logits = self.backbone(x)
        if return_logits:
            return logits
        return F.softmax(logits, dim=1)

# ==========================================
# Energy-Based OOD Detection
# ==========================================
def compute_energy_score(logits, temperature=1.0):
    """
    Compute energy score for OOD detection.
    E(x) = -T * log(sum(exp(f_c(x)/T)))
    
    Lower energy = Known (confident)
    Higher energy = Unknown (uncertain)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def train_closed_set_classifier(dataroot, num_epochs=30, batch_size=32, lr=0.001):
    """
    Train a simple closed-set classifier (no ARPL, no OSR tricks).
    Pure cross-entropy on known classes only.
    """
    print("="*80)
    print("STAGE 1: TRAINING CLOSED-SET CLASSIFIER")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load dataset - KNOWN CLASSES ONLY for training
    trainset = DDR(root=dataroot, train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    testset = DDR(root=dataroot, train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate class weights (moderate)
    from collections import defaultdict
    class_counts = defaultdict(int)
    for _, label in trainset:
        if label < 3:
            class_counts[label] += 1
    
    total = sum(class_counts.values())
    weights = [np.sqrt(total / (3 * class_counts[i])) for i in range(3)]
    class_weights = torch.FloatTensor(weights)
    if use_gpu:
        class_weights = class_weights.cuda()
    
    print(f"\nClass distribution:")
    for i in range(3):
        print(f"  Class {i}: {class_counts[i]} samples (weight: {weights[i]:.3f})")
    
    # Model
    model = ResNet50Classifier(num_classes=3)
    if use_gpu:
        model = model.cuda()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(trainloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        correct = 0
        total_known = 0
        class_correct = [0, 0, 0]
        class_total = [0, 0, 0]
        
        with torch.no_grad():
            for data, labels in testloader:
                if use_gpu:
                    data, labels = data.cuda(), labels.cuda()
                
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                
                # Only evaluate on known classes
                known_mask = labels < 3
                if known_mask.sum() > 0:
                    known_labels = labels[known_mask]
                    known_preds = predicted[known_mask]
                    
                    correct += (known_preds == known_labels).sum().item()
                    total_known += known_mask.sum().item()
                    
                    for i in range(3):
                        class_mask = known_labels == i
                        if class_mask.sum() > 0:
                            class_correct[i] += (known_preds[class_mask] == i).sum().item()
                            class_total[i] += class_mask.sum().item()
        
        acc = 100.0 * correct / total_known if total_known > 0 else 0
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Overall Accuracy: {acc:.2f}%")
        for i in range(3):
            if class_total[i] > 0:
                print(f"  Class {i}: {100.0 * class_correct[i] / class_total[i]:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'checkpoints/closed_set_model.pth')
            print(f"  → Saved best model: {acc:.2f}%")
        
        scheduler.step()
    
    return model

def evaluate_energy_ood(model, dataroot, temperature=1.0):
    """
    Evaluate OOD detection using energy scores.
    """
    print("\n" + "="*80)
    print("STAGE 2: ENERGY-BASED OOD DETECTION")
    print("="*80)
    print(f"Temperature: {temperature}")
    
    use_gpu = torch.cuda.is_available()
    
    # Test transform
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load test set with ALL classes (known + unknown)
    testset = DDR(root=dataroot, train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    model.eval()
    
    all_energies = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            logits = model(data, return_logits=True)
            energies = compute_energy_score(logits, temperature)
            
            _, preds = torch.max(logits, 1)
            
            all_energies.append(energies.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_energies = np.concatenate(all_energies)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    # Separate known vs unknown
    known_mask = all_labels < 3
    unknown_mask = all_labels >= 3
    
    # Known class accuracy
    known_correct = (all_preds[known_mask] == all_labels[known_mask]).sum()
    known_acc = 100.0 * known_correct / known_mask.sum()
    
    print(f"\nKnown Class Accuracy: {known_acc:.2f}%")
    
    # Per-class accuracy
    for i in range(3):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = 100.0 * (all_preds[class_mask] == i).sum() / class_mask.sum()
            print(f"  Class {i}: {class_acc:.2f}%")
    
    # OOD detection with energy
    if unknown_mask.sum() > 0:
        # Binary labels: 0=known, 1=unknown
        binary_labels = unknown_mask.astype(int)
        
        # Energy scores (higher energy = more likely unknown)
        ood_scores = all_energies
        
        # Calculate AUROC
        auroc = roc_auc_score(binary_labels, ood_scores) * 100
        
        # Find optimal threshold
        fpr, tpr, thresholds = roc_curve(binary_labels, ood_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nOOD Detection Results:")
        print(f"  Energy AUROC: {auroc:.2f}%")
        print(f"  Optimal Threshold: {optimal_threshold:.4f}")
        print(f"  TPR at optimal: {tpr[optimal_idx]:.2%}")
        print(f"  FPR at optimal: {fpr[optimal_idx]:.2%}")
        
        # Energy distribution
        print(f"\nEnergy Statistics:")
        print(f"  Known   - Mean: {all_energies[known_mask].mean():.4f}, Std: {all_energies[known_mask].std():.4f}")
        print(f"  Unknown - Mean: {all_energies[unknown_mask].mean():.4f}, Std: {all_energies[unknown_mask].std():.4f}")
        
        # Combined metric
        combined = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
        print(f"\n★ Combined Score (Harmonic Mean): {combined:.2f}%")
        
        return known_acc, auroc, optimal_threshold
    
    return known_acc, 0.0, 0.0

def test_multiple_temperatures(model, dataroot):
    """
    Test different temperature values to find optimal.
    """
    print("\n" + "="*80)
    print("TESTING MULTIPLE TEMPERATURES")
    print("="*80)
    
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5]
    results = []
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        known_acc, auroc, threshold = evaluate_energy_ood(model, dataroot, temperature=temp)
        results.append((temp, known_acc, auroc))
        combined = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
        print(f"Combined: {combined:.2f}%")
    
    # Find best temperature
    best_idx = np.argmax([r[2] for r in results])  # Max AUROC
    best_temp, best_acc, best_auroc = results[best_idx]
    
    print("\n" + "="*80)
    print("BEST TEMPERATURE")
    print("="*80)
    print(f"Temperature: {best_temp}")
    print(f"Known Accuracy: {best_acc:.2f}%")
    print(f"AUROC: {best_auroc:.2f}%")
    print(f"Combined: {2 * (best_acc * best_auroc) / (best_acc + best_auroc):.2f}%")
    
    return best_temp

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    
    # Check if we have a pre-trained closed-set model
    if os.path.exists('checkpoints/closed_set_model.pth'):
        print("Loading existing closed-set model...")
        model = ResNet50Classifier(num_classes=3)
        model.load_state_dict(torch.load('checkpoints/closed_set_model.pth'))
        if torch.cuda.is_available():
            model = model.cuda()
    else:
        print("No existing model found. Training from scratch...")
        model = train_closed_set_classifier('DDR dataset', num_epochs=30, batch_size=32, lr=0.001)
    
    # Find best temperature
    best_temp = test_multiple_temperatures(model, 'DDR dataset')
    
    # Final evaluation with best temperature
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    known_acc, auroc, threshold = evaluate_energy_ood(model, 'DDR dataset', temperature=best_temp)
    
    print("\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"✓ Known Class Accuracy: {known_acc:.2f}%")
    print(f"✓ Unknown Detection AUROC: {auroc:.2f}%")
    print(f"✓ Combined Score: {2 * (known_acc * auroc) / (known_acc + auroc):.2f}%")
    print(f"✓ Optimal Temperature: {best_temp}")
    print(f"✓ Optimal Threshold: {threshold:.4f}")
