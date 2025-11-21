"""
Comprehensive OOD Diagnosis
Check why Energy-based OOD is failing even with decent closed-set classifier
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x, return_logits=True):
        logits = self.backbone(x)
        if return_logits:
            return logits
        return F.softmax(logits, dim=1)

# Load model
model = ResNet50Classifier(num_classes=3)
if os.path.exists('checkpoints/focal_closed_set.pth'):
    model.load_state_dict(torch.load('checkpoints/focal_closed_set.pth'))
    print("✓ Loaded Focal Loss model")
elif os.path.exists('checkpoints/closed_set_model.pth'):
    model.load_state_dict(torch.load('checkpoints/closed_set_model.pth'))
    print("✓ Loaded standard model")
else:
    print("❌ No model found!")
    exit(1)

use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
model.eval()

# Load test data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset = DDR(root='DDR dataset', train=False, transform=transform,
              train_class_num=3, test_class_num=5, includes_all_train_class=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# Collect all predictions and scores
all_logits = []
all_labels = []
all_preds = []

with torch.no_grad():
    for data, labels in testloader:
        if use_gpu:
            data = data.cuda()
        
        logits = model(data)
        _, preds = torch.max(logits, 1)
        
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

# Masks
known_mask = all_labels < 3
unknown_mask = all_labels >= 3
binary_labels = unknown_mask.astype(int)

print("="*80)
print("COMPREHENSIVE OOD DIAGNOSIS")
print("="*80)

# 1. Check closed-set performance
print("\n1. CLOSED-SET CLASSIFIER PERFORMANCE:")
known_correct = (all_preds[known_mask] == all_labels[known_mask]).sum()
known_acc = 100.0 * known_correct / known_mask.sum()
print(f"   Overall: {known_acc:.2f}%")

for i in range(3):
    class_mask = all_labels == i
    if class_mask.sum() > 0:
        class_acc = 100.0 * (all_preds[class_mask] == i).sum() / class_mask.sum()
        print(f"   Class {i}: {class_acc:.2f}% ({class_mask.sum()} samples)")

# 2. Test multiple OOD scores
print("\n2. TESTING MULTIPLE OOD SCORES:")

# Compute all scores
probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
max_probs = probs.max(axis=1)
energies = -np.log(np.exp(all_logits).sum(axis=1))

def compute_auroc(scores, higher_is_unknown=True):
    """Compute AUROC, handling direction"""
    if not higher_is_unknown:
        scores = -scores
    try:
        return roc_auc_score(binary_labels, scores) * 100
    except:
        return 0.0

# Test different scores
scores_to_test = {
    'Energy (higher=unknown)': (energies, True),
    'Energy (lower=unknown)': (energies, False),
    'Max Softmax Prob (higher=known)': (max_probs, False),
    'Max Softmax Prob (lower=known)': (max_probs, True),
    'Negative Max Logit': (-all_logits.max(axis=1), True),
    'Entropy': (-(probs * np.log(probs + 1e-10)).sum(axis=1), True),
}

best_auroc = 0
best_method = ""

for name, (scores, direction) in scores_to_test.items():
    auroc = compute_auroc(scores, direction)
    print(f"   {name:40s}: {auroc:6.2f}%")
    
    if auroc > best_auroc:
        best_auroc = auroc
        best_method = name

print(f"\n   ★ Best Method: {best_method} ({best_auroc:.2f}%)")

# 3. Check prediction confidence on known vs unknown
print("\n3. CONFIDENCE ANALYSIS:")
print(f"   Max Prob - Known:   {max_probs[known_mask].mean():.4f} ± {max_probs[known_mask].std():.4f}")
print(f"   Max Prob - Unknown: {max_probs[unknown_mask].mean():.4f} ± {max_probs[unknown_mask].std():.4f}")
print(f"   Energy - Known:     {energies[known_mask].mean():.4f} ± {energies[known_mask].std():.4f}")
print(f"   Energy - Unknown:   {energies[unknown_mask].mean():.4f} ± {energies[unknown_mask].std():.4f}")

if max_probs[unknown_mask].mean() > max_probs[known_mask].mean():
    print("\n   ⚠️  PROBLEM: Unknown samples have HIGHER confidence than known!")
    print("       → Classifier is overconfident on unknowns")

if energies[unknown_mask].mean() < energies[known_mask].mean():
    print("\n   ⚠️  PROBLEM: Unknown samples have LOWER energy than known!")
    print("       → Energy-based OOD is inverted")

# 4. Check what unknowns are being predicted as
print("\n4. UNKNOWN PREDICTIONS:")
unknown_preds = all_preds[unknown_mask]
pred_counts = np.bincount(unknown_preds, minlength=3)
total_unknown = len(unknown_preds)

for i in range(3):
    pct = 100.0 * pred_counts[i] / total_unknown
    print(f"   Predicted as Class {i}: {pred_counts[i]:5d} ({pct:5.1f}%)")

# 5. Recommendation
print("\n" + "="*80)
print("DIAGNOSIS & RECOMMENDATIONS:")
print("="*80)

if known_acc < 85:
    print("\n❌ PRIMARY ISSUE: Closed-set classifier is too weak")
    print(f"   Current accuracy: {known_acc:.2f}% (need ≥88%)")
    print("\n   SOLUTIONS:")
    print("   1. Train longer (more epochs)")
    print("   2. Check if Focal Loss helped Class 1")
    print("   3. Use stronger augmentation")
    
elif best_auroc < 60:
    print("\n❌ PRIMARY ISSUE: OOD detection completely failing")
    print(f"   Best AUROC: {best_auroc:.2f}% (need ≥80%)")
    print("\n   POSSIBLE CAUSES:")
    print("   • Classifier is overconfident on unknowns")
    print("   • DR grades overlap too much (No_DR → Mild → Moderate)")
    print("   • Unknowns (Severe, Proliferative) are IN-distribution")
    print("\n   SOLUTIONS:")
    print("   1. Use Mahalanobis distance (feature space method)")
    print("   2. Train with outlier exposure")
    print("   3. Use ODIN (temperature + perturbation)")
    print("   4. Accept that DR grades may not be separable as OOD task")
    
elif best_auroc >= 80:
    print(f"\n✅ SOLUTION FOUND: {best_method}")
    print(f"   AUROC: {best_auroc:.2f}%")
    print(f"   Known Acc: {known_acc:.2f}%")
    print(f"   Combined: {2 * (known_acc * best_auroc) / (known_acc + best_auroc):.2f}%")
else:
    print(f"\n⚠️  PARTIAL SUCCESS: {best_method}")
    print(f"   AUROC: {best_auroc:.2f}% (moderate)")
    print("   May need advanced methods (Mahalanobis, ODIN)")

print("\n" + "="*80)
