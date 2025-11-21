import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project path
sys.path.append(".")

from datasets import DDR
from Utils.medical_augmentations import get_ddr_transforms
from Utils.losses import FocalLoss
from backbones.efficientnet_wrapper import efficientnet_b0

class OSRModel(nn.Module):
    def __init__(self, num_classes, backbone_name='efficientnet_b0'):
        super(OSRModel, self).__init__()
        
        # Initialize backbone
        if backbone_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0(num_classes=num_classes, backbone_fc=False)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Backbone {backbone_name} not supported in this script yet.")
            
        # Projection head (optional, but good for OSR)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.projection(features)
        logits = self.classifier(embeddings)
        return embeddings, logits

def train(args, model, train_loader, criterion, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        _, logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} Acc: {100.*correct/total:.2f}% '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                  
    if scheduler is not None:
        scheduler.step()
        
    return running_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, device, known_classes):
    model.eval()
    all_logits = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _, logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())
            
    all_logits = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)
    
    # 1. Closed Set Accuracy (on known classes only)
    known_mask = all_targets < len(known_classes)
    known_logits = all_logits[known_mask]
    known_targets = all_targets[known_mask]
    
    _, known_preds = known_logits.max(1)
    closed_set_acc = known_preds.eq(known_targets).float().mean().item() * 100
    
    # 2. Unknown Detection (AUROC)
    # Simple baseline: Use max softmax probability as confidence
    # Unknowns should have LOWER max probability
    max_probs, _ = all_probs.max(1)
    
    # Create binary labels: 0 for known, 1 for unknown
    is_unknown = (all_targets >= len(known_classes)).float()
    
    # Score for being unknown = 1 - max_prob (or negative max_prob)
    unknown_scores = 1.0 - max_probs
    
    try:
        auroc = roc_auc_score(is_unknown, unknown_scores)
    except ValueError:
        auroc = 0.5 # Handle case where only one class is present
        
    return {
        'closed_set_acc': closed_set_acc,
        'auroc': auroc,
        'logits': all_logits,
        'targets': all_targets,
        'probs': all_probs
    }

def main():
    parser = argparse.ArgumentParser(description='Improved DDR OSR Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_class_num', type=int, default=3, help='Number of known classes')
    parser.add_argument('--test_class_num', type=int, default=5, help='Total classes')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--output_dir', type=str, default='checkpoints/ddr_improved', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data Augmentation & Loading
    print("Preparing data with medical augmentations...")
    train_transform = get_ddr_transforms(is_training=True)
    test_transform = get_ddr_transforms(is_training=False)
    
    trainset = DDR(root='./DDR dataset', train=True, transform=train_transform,
                   train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                   includes_all_train_class=True)
    
    testset = DDR(root='./DDR dataset', train=False, transform=test_transform,
                  train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                  includes_all_train_class=True)
                  
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")
    
    # 2. Model Setup (EfficientNet-B0)
    print("Initializing EfficientNet-B0 model...")
    model = OSRModel(num_classes=args.train_class_num, backbone_name='efficientnet_b0')
    model = model.to(device)
    
    # 3. Loss Function (Focal Loss)
    # Calculate class weights for Focal Loss
    print("Calculating class weights...")
    targets = np.array(trainset.targets)
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    # Inverse frequency weights
    weights = total_samples / (len(class_counts) * class_counts)
    weights = torch.FloatTensor(weights).to(device)
    
    print(f"Class counts: {class_counts}")
    print(f"Class weights: {weights.cpu().numpy()}")
    
    criterion = FocalLoss(alpha=weights.tolist(), gamma=args.gamma)
    
    # 4. Optimizer & Scheduler (Cosine Annealing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training Loop
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, model, train_loader, criterion, optimizer, scheduler, device, epoch)
        
        results = evaluate(model, test_loader, device, range(args.train_class_num))
        
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Test Closed Set Acc: {results['closed_set_acc']:.2f}%")
        print(f"  Unknown Detection AUROC: {results['auroc']:.4f}")
        
        # Save best model
        if results['closed_set_acc'] > best_acc:
            best_acc = results['closed_set_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': best_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print("  -> Saved new best model")
            
    print("Training complete!")
    print(f"Best Closed Set Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
