import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime

# Add project path
sys.path.append(".")

from datasets import DDR
from Utils.medical_augmentations import get_ddr_transforms
from backbones.efficientnet_wrapper import efficientnet_b0

# ==========================================
# DFP Components
# ==========================================

class DFPNet(nn.Module):
    def __init__(self, num_classes=3, embed_dim=512):
        super(DFPNet, self).__init__()
        # Backbone
        self.backbone = efficientnet_b0(num_classes=num_classes, backbone_fc=False)
        self.feat_dim = 1280 # EfficientNet-B0 output
        
        # Embedding Layer (Projection Head)
        self.embedding = nn.Sequential(
            nn.Linear(self.feat_dim, embed_dim),
            nn.PReLU()
        )
        
        # Classifier (for Cross Entropy)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding(features)
        logits = self.classifier(embeddings)
        return logits, embeddings

class DFPLoss(nn.Module):
    """
    Deep Feature Preserving Loss
    Combines Cross Entropy Loss with Distance Loss (Intra-class compactness)
    """
    def __init__(self, num_classes, feat_dim, alpha=1.0):
        super(DFPLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha # Weight for distance loss
        
        # Centroids (Learnable parameters)
        self.centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centroids)
        
        self.ce_criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, embeddings, targets):
        # 1. Cross Entropy Loss
        ce_loss = self.ce_criterion(logits, targets)
        
        # 2. Distance Loss
        # Get centroids for the target classes
        batch_centroids = self.centroids[targets]
        
        # Calculate distance between embeddings and their class centroids
        # L2 distance squared
        distances = torch.sum((embeddings - batch_centroids) ** 2, dim=1)
        distance_loss = 0.5 * distances.mean()
        
        # Total loss
        total_loss = ce_loss + self.alpha * distance_loss
        
        return total_loss, ce_loss, distance_loss

# ==========================================
# Training & Evaluation
# ==========================================

def train(model, criterion, train_loader, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    running_ce = 0.0
    running_dist = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits, embeddings = model(inputs)
        
        loss, ce_loss, dist_loss = criterion(logits, embeddings, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_ce += ce_loss.item()
        running_dist += dist_loss.item()
        
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} (CE: {ce_loss.item():.4f}, Dist: {dist_loss.item():.4f}) '
                  f'Acc: {100.*correct/total:.2f}%')
                  
    return running_loss / len(train_loader), 100. * correct / total

def evaluate_dfp(model, criterion, test_loader, device, known_classes):
    model.eval()
    all_distances = []
    all_targets = []
    
    centroids = criterion.centroids.detach() # (num_classes, embed_dim)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, embeddings = model(inputs)
            
            # Calculate distance to NEAREST centroid for each sample
            # embeddings: (B, D)
            # centroids: (K, D)
            
            # Expand for broadcasting
            # (B, 1, D) - (1, K, D) -> (B, K, D)
            diff = embeddings.unsqueeze(1) - centroids.unsqueeze(0)
            dists = torch.sum(diff ** 2, dim=2) # (B, K)
            
            # Get minimum distance (distance to nearest known class)
            min_dists, _ = dists.min(dim=1)
            
            all_distances.append(min_dists.cpu())
            all_targets.append(targets.cpu())
            
    all_distances = torch.cat(all_distances)
    all_targets = torch.cat(all_targets)
    
    # Create binary labels: 0 for known, 1 for unknown
    is_unknown = (all_targets >= len(known_classes)).float()
    
    # For AUROC: Higher score should indicate Positive class (Unknown)
    # Unknowns should have LARGER distances
    # So we can use distances directly as scores
    
    try:
        auroc = roc_auc_score(is_unknown, all_distances)
    except ValueError:
        auroc = 0.5
        
    return auroc, all_distances, is_unknown

def main():
    parser = argparse.ArgumentParser(description='DDR DFP Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='Weight for distance loss')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--output_dir', type=str, default='checkpoints/ddr_dfp', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    print("Preparing DDR dataset...")
    train_transform = get_ddr_transforms(is_training=True)
    test_transform = get_ddr_transforms(is_training=False)
    
    trainset = DDR(root='./DDR dataset', train=True, transform=train_transform,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root='./DDR dataset', train=False, transform=test_transform,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
                  
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model & Loss
    print("Initializing DFP Model...")
    model = DFPNet(num_classes=3, embed_dim=args.embed_dim).to(device)
    criterion = DFPLoss(num_classes=3, feat_dim=args.embed_dim, alpha=args.alpha).to(device)
    
    # Optimizer (optimize both model and centroids)
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters()} # Centroids are here
    ], lr=args.lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_auroc = 0.0
    
    print("Starting DFP Training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, criterion, train_loader, optimizer, device, epoch)
        scheduler.step()
        
        auroc, dists, labels = evaluate_dfp(model, criterion, test_loader, device, range(3))
        
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Unknown Detection AUROC: {auroc:.4f}")
        
        if auroc > best_auroc:
            best_auroc = auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'centroids': criterion.centroids,
                'auroc': best_auroc
            }, os.path.join(args.output_dir, 'best_dfp_model.pth'))
            print("  -> Saved new best model")
            
    print(f"Training Complete. Best AUROC: {best_auroc:.4f}")

if __name__ == '__main__':
    main()
