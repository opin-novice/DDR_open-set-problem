"""
QUICK OOD TEST - Multiple Methods
"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms, torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)

model = ResNet50Classifier(3)
model.load_state_dict(torch.load('checkpoints/focal_closed_set.pth'))
model = model.cuda() if torch.cuda.is_available() else model
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

testset = DDR('DDR dataset', train=False, transform=transform, train_class_num=3, test_class_num=5, includes_all_train_class=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

all_logits, all_labels, all_preds = [], [], []
use_gpu = torch.cuda.is_available()

with torch.no_grad():
    for data, labels in testloader:
        if use_gpu: data = data.cuda()
        logits = model(data)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_preds.append(torch.max(logits, 1)[1].cpu().numpy())

all_logits = np.concatenate(all_logits)
all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

known_mask = all_labels < 3
unknown_mask = all_labels >= 3

# Closed-set accuracy
known_acc = 100.0 * (all_preds[known_mask] == all_labels[known_mask]).sum() / known_mask.sum()
print(f"Known Accuracy: {known_acc:.2f}%")
for i in range(3):
    mask = all_labels == i
    if mask.sum() > 0:
        print(f"  Class {i}: {100.0 * (all_preds[mask] == i).sum() / mask.sum():.2f}%")

# OOD scores
probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=1, keepdims=True)
max_probs = probs.max(axis=1)
energies = -np.log(np.exp(all_logits).sum(axis=1))
max_logits = all_logits.max(axis=1)

binary = unknown_mask.astype(int)

print("\nOOD AUROC Results:")
print(f"  MSP (1-maxprob):        {roc_auc_score(binary, 1-max_probs)*100:.2f}%")
print(f"  Energy:                 {roc_auc_score(binary, energies)*100:.2f}%")
print(f"  Max Logit (negative):   {roc_auc_score(binary, -max_logits)*100:.2f}%")

print("\nConfidence Stats:")
print(f"  Known   MaxProb: {max_probs[known_mask].mean():.3f}") 
print(f"  Unknown MaxProb: {max_probs[unknown_mask].mean():.3f}")
print(f"  Known   Energy:  {energies[known_mask].mean():.3f}")
print(f"  Unknown Energy:  {energies[unknown_mask].mean():.3f}")

if max_probs[unknown_mask].mean() >= max_probs[known_mask].mean():
    print("\nPROBLEM: Unknowns have HIGHER confidence - overconfident classifier")
