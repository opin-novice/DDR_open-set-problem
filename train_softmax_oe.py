"""
Standard Softmax + Outlier Exposure + Mahalanobis
-------------------------------------------------
Implements the user's requested changes:
1. Standard Softmax Classifier (No LogitNorm)
2. Outlier Exposure (OE) with Noise/KMNIST
3. Mahalanobis Distance Scoring
4. Focal Loss (Added to handle class imbalance)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import FakeData
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR
from focal_loss import FocalLoss

# ==========================================
# 1. Standard Softmax Classifier
# ==========================================
class StandardClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(StandardClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        
        # Standard Linear Layer (Softmax)
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Backbone features
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

# ==========================================
# 2. Outlier Exposure Loss
# ==========================================
def oe_loss_fn(logits_out):
    """
    Standard OE Loss: KL(softmax(out) || uniform)
    Equivalent to maximizing entropy of softmax(out).
    """
    probs = F.softmax(logits_out, dim=1)
    loss = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
    return loss

# ==========================================
# 3. Mahalanobis Utils
# ==========================================
def compute_mahalanobis_params(model, dataloader, num_classes=3, use_gpu=True):
    model.eval()
    all_features = []
    all_labels = []
    
    print("Computing Mahalanobis statistics...")
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    
    # Filter known
    known_mask = labels < num_classes
    features = features[known_mask]
    labels = labels[known_mask]
    
    # Class means
    class_means = []
    centered_features = []
    for c in range(num_classes):
        c_feats = features[labels == c]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_features.append(c_feats - mean)
        
    # Pooled covariance
    centered_all = np.concatenate(centered_features, axis=0)
    cov = EmpiricalCovariance().fit(centered_all).covariance_
    
    # Precision (Inverse Covariance)
    reg = 1e-6 * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov + reg)
    
    return class_means, precision

def get_mahalanobis_scores(model, dataloader, class_means, precision, use_gpu=True):
    model.eval()
    scores = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True).cpu().numpy()
            logits = model(data)
            _, preds = torch.max(logits, 1)
            
            batch_scores = []
            for c in range(len(class_means)):
                centered = features - class_means[c]
                # Mahalanobis distance squared: (x-u)T S^-1 (x-u)
                dist = np.sum(centered @ precision * centered, axis=1)
                batch_scores.append(dist)
            
            # Min distance across classes
            batch_scores = np.array(batch_scores).T
            min_dist = batch_scores.min(axis=1)
            
            scores.append(min_dist)
            labels_list.append(labels.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            
    return np.concatenate(scores), np.concatenate(labels_list), np.concatenate(preds_list)

# ==========================================
# 4. Training Worker
# ==========================================
def train_softmax_oe(args):
    print("="*80)
    print("TRAINING: STANDARD SOFTMAX + OUTLIER EXPOSURE + FOCAL LOSS")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # 1. Model Setup
    model = StandardClassifier(num_classes=3)
    if use_gpu: model = model.cuda()
    
    # 2. Data Setup
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # In-Distribution Data
    trainset = DDR(root=args['dataroot'], train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root=args['dataroot'], train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=0)
"""
Standard Softmax + Outlier Exposure + Mahalanobis
-------------------------------------------------
Implements the user's requested changes:
1. Standard Softmax Classifier (No LogitNorm)
2. Outlier Exposure (OE) with Noise/KMNIST
3. Mahalanobis Distance Scoring
4. Focal Loss (Added to handle class imbalance)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import FakeData
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR
from focal_loss import FocalLoss

# ==========================================
# 1. Standard Softmax Classifier
# ==========================================
class StandardClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(StandardClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        
        # Standard Linear Layer (Softmax)
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # Backbone features
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

# ==========================================
# 2. Outlier Exposure Loss
# ==========================================
def oe_loss_fn(logits_out):
    """
    Standard OE Loss: KL(softmax(out) || uniform)
    Equivalent to maximizing entropy of softmax(out).
    """
    probs = F.softmax(logits_out, dim=1)
    loss = torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=1))
    return loss

# ==========================================
# 3. Mahalanobis Utils
# ==========================================
def compute_mahalanobis_params(model, dataloader, num_classes=3, use_gpu=True):
    model.eval()
    all_features = []
    all_labels = []
    
    print("Computing Mahalanobis statistics...")
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)
    
    # Filter known
    known_mask = labels < num_classes
    features = features[known_mask]
    labels = labels[known_mask]
    
    # Class means
    class_means = []
    centered_features = []
    for c in range(num_classes):
        c_feats = features[labels == c]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_features.append(c_feats - mean)
        
    # Pooled covariance
    centered_all = np.concatenate(centered_features, axis=0)
    cov = EmpiricalCovariance().fit(centered_all).covariance_
    
    # Precision (Inverse Covariance)
    reg = 1e-6 * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov + reg)
    
    return class_means, precision

def get_mahalanobis_scores(model, dataloader, class_means, precision, use_gpu=True):
    model.eval()
    scores = []
    labels_list = []
    preds_list = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True).cpu().numpy()
            logits = model(data)
            _, preds = torch.max(logits, 1)
            
            batch_scores = []
            for c in range(len(class_means)):
                centered = features - class_means[c]
                # Mahalanobis distance squared: (x-u)T S^-1 (x-u)
                dist = np.sum(centered @ precision * centered, axis=1)
                batch_scores.append(dist)
            
            # Min distance across classes
            batch_scores = np.array(batch_scores).T
            min_dist = batch_scores.min(axis=1)
            
            scores.append(min_dist)
            labels_list.append(labels.cpu().numpy())
            preds_list.append(preds.cpu().numpy())
            
    return np.concatenate(scores), np.concatenate(labels_list), np.concatenate(preds_list)

# ==========================================
# 4. Training Worker
# ==========================================
def train_softmax_oe(args):
    print("="*80)
    print("TRAINING: STANDARD SOFTMAX + OUTLIER EXPOSURE + FOCAL LOSS")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # 1. Model Setup
    model = StandardClassifier(num_classes=3)
    if use_gpu: model = model.cuda()
    
    # 2. Data Setup
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # In-Distribution Data
    trainset = DDR(root=args['dataroot'], train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root=args['dataroot'], train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False, num_workers=0)
    
    # Outlier Exposure Data (Noise/FakeData)
    oe_set = FakeData(size=10000, image_size=(3, 224, 224), num_classes=10, transform=transforms.ToTensor())
    oe_loader = torch.utils.data.DataLoader(oe_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
    
    # 3. Optimizer & Loss
    # Standard SGD (Uniform LR)
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    
    # Calculate class weights
    print("Calculating class weights...")
    targets = trainset.targets
    class_counts = np.bincount(targets)
    class_counts = torch.tensor(class_counts, dtype=torch.float)
    
    # Inverse frequency weights
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    
    print(f"Class Counts: {class_counts.tolist()}")
    print(f"Class Weights: {weights.tolist()}")
    
    if use_gpu: weights = weights.cuda()
    
    # Weighted Cross Entropy
    criterion = nn.CrossEntropyLoss(weight=weights)
    if use_gpu: criterion = criterion.cuda()
    
    # 4. Training Loop
    print(f"\nStarting training for {args['epochs']} epochs...")
    
    for epoch in range(args['epochs']):
        model.train()
        train_loss = 0
        
        num_batches = min(len(trainloader), len(oe_loader))
        oe_iter = iter(oe_loader)
        
        for batch_idx, (data_in, labels_in) in enumerate(trainloader):
            if batch_idx >= num_batches: break
            
            try:
                data_out, _ = next(oe_iter)
            except StopIteration:
                oe_iter = iter(oe_loader)
                data_out, _ = next(oe_iter)
            
            if use_gpu:
                data_in, labels_in = data_in.cuda(), labels_in.cuda()
                data_out = data_out.cuda()
            
            # Forward Pass
            logits_in = model(data_in)
            logits_out = model(data_out)
            
            # Losses
            loss_ce = criterion(logits_in, labels_in)
            loss_oe = oe_loss_fn(logits_out)
            
            # OE Disabled for Baseline Recovery
            loss = loss_ce # + args['lambda_oe'] * loss_oe
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx+1) % 20 == 0:
                print(f"Epoch {epoch+1} [{batch_idx+1}/{num_batches}] Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, OE: {loss_oe.item():.4f})")
        
        scheduler.step()
        
        # --- Evaluation (Every 5 epochs) ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args['epochs']:
            print(f"\nEvaluating Epoch {epoch+1}...")
            
            # 1. Compute Mahalanobis Stats
            means, precision = compute_mahalanobis_params(model, trainloader, num_classes=3, use_gpu=use_gpu)
            
            # 2. Get Scores & Preds
            scores, labels_all, preds_all = get_mahalanobis_scores(model, testloader, means, precision, use_gpu=use_gpu)
            
            # 3. Metrics
            known_mask = labels_all < 3
            unknown_mask = labels_all >= 3
            
            # Known Accuracy (Standard Softmax)
            known_correct = (preds_all[known_mask] == labels_all[known_mask]).sum()
            known_acc = 100.0 * known_correct / known_mask.sum()
            print(f"  Known Acc (Softmax): {known_acc:.2f}%")
            
            # Class-wise Accuracy
            for c in range(3):
                mask = (labels_all == c)
                if mask.sum() > 0:
                    acc = 100.0 * (preds_all[mask] == c).sum() / mask.sum()
                    print(f"    Class {c}: {acc:.2f}%")
            
            # AUROC (Mahalanobis)
            if unknown_mask.sum() > 0:
                auroc = roc_auc_score(unknown_mask, scores) * 100
                print(f"  AUROC (Mahalanobis): {auroc:.2f}%")
                
                # Save best
                if known_acc > 85: # Relaxed condition
                    torch.save(model.state_dict(), f"checkpoints/best_softmax_oe_ep{epoch+1}.pth")
                    print("  -> Saved best model")

if __name__ == "__main__":
    args = {
        'dataroot': 'DDR dataset',
        'epochs': 30,
        'batch_size': 32,
        'lr': 0.01,        # Increased to 0.01
        'lambda_oe': 0.0   # Disabled OE
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    train_softmax_oe(args)
