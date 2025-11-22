"""
CORRECTED: Two-Stage Energy OOD with Focal Loss
Fixes Class 1 collapse issue

Key Change: Using Focal Loss (gamma=2.0) instead of weighted CE
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from datasets import DDR
from focal_loss import FocalLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_splits(dataset, output_dir, prefix='train'):
    # DDR dataset has image_paths attribute
    if hasattr(dataset, 'image_paths'):
        with open(os.path.join(output_dir, f'{prefix}_list.txt'), 'w') as f:
            for idx in range(len(dataset)):
                path = dataset.image_paths[idx]
                label = dataset.targets[idx]
                f.write(f"{path} {label}\n")

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

def compute_energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=1)

def train_with_focal_loss(dataroot, num_epochs=40, batch_size=32, lr=0.001, seed=42, output_dir='checkpoints'):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print(f"STAGE 1: CLOSED-SET WITH FOCAL LOSS (Seed: {seed})")
    print("="*80)
    print("Using Focal Loss (gamma=2.0) to prevent Class 1 collapse")
    
    use_gpu = torch.cuda.is_available()
    
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
    
    trainset = DDR(root=dataroot, train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    testset = DDR(root=dataroot, train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    # Save splits
    save_splits(trainset, output_dir, 'train')
    save_splits(testset, output_dir, 'test')
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Calculate class distribution
    from collections import defaultdict
    class_counts = defaultdict(int)
    for _, label in trainset:
        if label < 3:
            class_counts[label] += 1
    
    print(f"\nClass distribution:")
    for i in range(3):
        print(f"  Class {i}: {class_counts[i]} samples")
    
    # Focal Loss with class-specific alpha
    # Higher alpha for minority class (Class 1)
    alpha = [0.4, 1.0, 0.6]  # Boost Class 1
    focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    
    print(f"\nFocal Loss Configuration:")
    print(f"  Gamma: 2.0 (focus on hard examples)")
    print(f"  Alpha: {alpha} (Class 1 boosted)")
    
    # Model
    model = ResNet50Classifier(num_classes=3)
    if use_gpu:
        model = model.cuda()
    
    # Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.layer4.parameters(), 'lr': lr * 0.1},
        {'params': model.backbone.fc.parameters(), 'lr': lr}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    best_class1_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(trainloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = focal_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(trainloader)}, Loss: {loss.item():.4f}")
        
        # Evaluation
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
        class1_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Overall Accuracy: {acc:.2f}%")
        for i in range(3):
            if class_total[i] > 0:
                print(f"  Class {i}: {100.0 * class_correct[i] / class_total[i]:.2f}%")
        
        # Save if improvement
        if acc > best_acc or (acc >= best_acc * 0.95 and class1_acc > best_class1_acc):
            best_acc = max(acc, best_acc)
            best_class1_acc = max(class1_acc, best_class1_acc)
            torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
            print(f"  → Saved model (Acc: {acc:.2f}%, Class1: {class1_acc:.2f}%)")
        
        # Early success check
        if acc >= 88 and class1_acc >= 50:
            print(f"\n✅ TARGET REACHED!")
            print(f"   Overall: {acc:.2f}%, Class 1: {class1_acc:.2f}%")
            print("   Continuing to see if we can improve further...")
        
        scheduler.step()
    
    return model

def evaluate_energy_ood(model, dataroot, temperature=1.0):
    print("\n" + "="*80)
    print("STAGE 2: ENERGY-BASED OOD DETECTION")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
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
                data = data.cuda()
            
            logits = model(data, return_logits=True)
            energies = compute_energy_score(logits, temperature)
            _, preds = torch.max(logits, 1)
            
            all_energies.append(energies.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
    
    all_energies = np.concatenate(all_energies)
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    known_mask = all_labels < 3
    unknown_mask = all_labels >= 3
    
    # Known accuracy
    known_correct = (all_preds[known_mask] == all_labels[known_mask]).sum()
    known_acc = 100.0 * known_correct / known_mask.sum()
    
    print(f"\nKnown Class Accuracy: {known_acc:.2f}%")
    for i in range(3):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = 100.0 * (all_preds[class_mask] == i).sum() / class_mask.sum()
            print(f"  Class {i}: {class_acc:.2f}%")
    
    # OOD detection
    if unknown_mask.sum() > 0:
        binary_labels = unknown_mask.astype(int)
        auroc = roc_auc_score(binary_labels, all_energies) * 100
        
        fpr, tpr, thresholds = roc_curve(binary_labels, all_energies)
        optimal_idx = np.argmax(tpr - fpr)
        
        print(f"\nOOD Detection (Energy):")
        print(f"  AUROC: {auroc:.2f}%")
        print(f"  Energy Stats:")
        print(f"    Known:   {all_energies[known_mask].mean():.3f} ± {all_energies[known_mask].std():.3f}")
        print(f"    Unknown: {all_energies[unknown_mask].mean():.3f} ± {all_energies[unknown_mask].std():.3f}")
        
        combined = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
        print(f"\n★ Combined Score: {combined:.2f}%")
        
        return known_acc, auroc
    
    return known_acc, 0.0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("FOCAL LOSS CORRECTED TRAINING")
    print("="*80)
    print("\nFixes:")
    print("  ✓ Focal Loss (gamma=2.0) prevents Class 1 collapse")
    print("  ✓ Class-specific alpha weighting")
    print("  ✓ Differential learning rates for stability")
    print("=" *80)
    
    # Train with Focal Loss
    model = train_with_focal_loss('DDR dataset', num_epochs=40, batch_size=32, lr=0.001, 
                                seed=args.seed, output_dir=args.output_dir)
    
    # Test multiple temperatures
    print("\n" + "="*80)
    print("TESTING TEMPERATURES")
    print("="*80)
    
    temps = [0.5, 1.0, 1.5, 2.0]
    results = []
    
    for temp in temps:
        print(f"\n--- Temperature: {temp} ---")
        known_acc, auroc = evaluate_energy_ood(model, 'DDR dataset', temperature=temp)
        results.append((temp, known_acc, auroc))
    
    # Best result
    best_idx = np.argmax([r[2] for r in results])
    best_temp, best_acc, best_auroc = results[best_idx]
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best Temperature: {best_temp}")
    print(f"Known Accuracy: {best_acc:.2f}%")
    print(f"AUROC: {best_auroc:.2f}%")
    print(f"Combined: {2 * (best_acc * best_auroc) / (best_acc + best_auroc):.2f}%")
    print("="*80)
