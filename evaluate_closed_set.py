"""
Quick evaluation of saved closed-set model to verify Class 1 collapse issue
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Load model
model = ResNet50Classifier(num_classes=3)
model.load_state_dict(torch.load('checkpoints/closed_set_model.pth'))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

# Load test data
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset = DDR(root='DDR dataset', train=False, transform=transform_test,
              train_class_num=3, test_class_num=5, includes_all_train_class=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

# Evaluate
all_preds = []
all_labels = []
all_logits = []

use_gpu = torch.cuda.is_available()

with torch.no_grad():
    for data, labels in testloader:
        if use_gpu:
            data = data.cuda()
        
        logits = model(data)
        _, preds = torch.max(logits, 1)
        
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_logits = np.concatenate(all_logits)

# Known class accuracy
known_mask = all_labels < 3
known_preds = all_preds[known_mask]
known_labels = all_labels[known_mask]

overall_acc = 100.0 * (known_preds == known_labels).sum() / len(known_labels)

print("="*80)
print("CLOSED-SET CLASSIFIER EVALUATION")
print("="*80)
print(f"\nOverall Known Class Accuracy: {overall_acc:.2f}%")

# Per-class
print("\nPer-Class Performance:")
for i in range(3):
    class_mask = known_labels == i
    if class_mask.sum() > 0:
        class_correct = (known_preds[class_mask] == i).sum()
        class_total = class_mask.sum()
        class_acc = 100.0 * class_correct / class_total
        print(f"  Class {i}: {class_acc:.2f}% ({class_correct}/{class_total})")
        
        # Check prediction distribution for this class
        unique, counts = np.unique(known_preds[class_mask], return_counts=True)
        print(f"    Predictions: {dict(zip(unique, counts))}")

# Confusion matrix for known classes
print("\nConfusion Matrix (Known Classes):")
print("          Pred 0  Pred 1  Pred 2")
for true_class in range(3):
    class_mask = known_labels == true_class
    if class_mask.sum() > 0:
        preds_for_class = known_preds[class_mask]
        counts = [
            (preds_for_class == 0).sum(),
            (preds_for_class == 1).sum(),
            (preds_for_class == 2).sum()
        ]
        print(f"True {true_class}:  {counts[0]:6d}  {counts[1]:6d}  {counts[2]:6d}")

# Check logit statistics
print("\nLogit Statistics (Known Classes):")
for i in range(3):
    class_mask = known_labels == i
    if class_mask.sum() > 0:
        class_logits = all_logits[known_mask][class_mask]
        print(f"  Class {i}:")
        print(f"    Logit 0: mean={class_logits[:, 0].mean():.3f}, std={class_logits[:, 0].std():.3f}")
        print(f"    Logit 1: mean={class_logits[:, 1].mean():.3f}, std={class_logits[:, 1].std():.3f}")
        print(f"    Logit 2: mean={class_logits[:, 2].mean():.3f}, std={class_logits[:, 2].std():.3f}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

class_1_acc = 100.0 * (known_preds[known_labels == 1] == 1).sum() / (known_labels == 1).sum()
if class_1_acc < 10:
    print("❌ CLASS 1 COLLAPSE CONFIRMED")
    print(f"   Class 1 accuracy: {class_1_acc:.2f}% (< 10%)")
    print("   → Classifier is broken for minority class")
    print("   → Energy OOD will NOT work with this classifier")
    print("\nRECOMMENDATION:")
    print("  Use Focal Loss with gamma=2.0 to prevent collapse")
elif overall_acc < 85:
    print("⚠️  CLASSIFIER UNDERPERFORMING")
    print(f"   Overall accuracy: {overall_acc:.2f}% (< 85%)")
    print("   → Need better closed-set performance for good OOD")
else:
    print("✅ CLASSIFIER LOOKS GOOD")
    print(f"   Overall: {overall_acc:.2f}%")
    print(f"   Class 1: {class_1_acc:.2f}%")
