"""
Balanced ARPL Training for DDR Dataset
Optimized for BOTH known class accuracy AND unknown detection

Key improvements over previous version:
1. Moderate class weighting (not too aggressive)
2. Higher ARPL margin weight for better unknown detection
3. Optimize for combined metric (harmonic mean)
4. Better learning rate schedule
"""

import os
import argparse
import datetime
import time
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'OSR', 'ARPL'))

from datasets import DDR
from OSR.ARPL.loss.ARPLoss import ARPLoss
from OSR.ARPL.utils import AverageMeter

# ==========================================
# ResNet50 Wrapper for ARPL
# ==========================================
class ResNet50_ARPL(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128, use_gpu=True):
        super(ResNet50_ARPL, self).__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final FC layer
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # Projection head for ARPL
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        
    def forward(self, x, return_feature=False):
        # Extract features from backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Project to lower dimension for ARPL
        features = self.projection(x)
        
        # Normalize features to unit length (Critical for distance-based OSR)
        features = F.normalize(features, p=2, dim=1)
        
        return features, features

# ==========================================
# Training Function
# ==========================================
def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        # Forward pass
        features, _ = net(data, return_feature=True)
        
        # ARPLoss forward
        logits, loss = criterion(features, features, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("  Batch {}/{}\t Loss {:.4f}".format(batch_idx+1, len(trainloader), losses.val))

    return losses.avg

# ==========================================
# Evaluation Function
# ==========================================
def evaluate_detailed(net, criterion, testloader, known_classes, options):
    net.eval()
    
    all_preds = []
    all_targets = []
    all_scores = []
    all_dists = []
    
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            features, _ = net(data, return_feature=True)
            
            dist_dot_p = criterion.Dist(features, center=criterion.points, metric='dot')
            dist_l2_p = criterion.Dist(features, center=criterion.points)
            logits = dist_l2_p - dist_dot_p
            
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_scores.append(max_probs.cpu().numpy())
            all_dists.append(torch.min(dist_l2_p, dim=1)[0].cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)
    all_dists = np.concatenate(all_dists)
    
    # Known Accuracy
    known_mask = all_targets < len(known_classes)
    unknown_mask = all_targets >= len(known_classes)
    
    if np.sum(known_mask) > 0:
        known_preds = all_preds[known_mask]
        known_targets_subset = all_targets[known_mask]
        known_acc = np.mean(known_preds == known_targets_subset) * 100
        
        # Per-class accuracy
        class_accs = []
        for c in range(len(known_classes)):
            class_mask = known_targets_subset == c
            if np.sum(class_mask) > 0:
                class_acc = np.mean(known_preds[class_mask] == c) * 100
                class_accs.append(class_acc)
    else:
        known_acc = 0.0
        class_accs = []
        
    # Unknown Detection (AUROC)
    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        binary_labels = unknown_mask.astype(int)
        
        # Use distance-based score (best for ARPL)
        unknown_scores_dist = all_dists
        auroc = roc_auc_score(binary_labels, unknown_scores_dist) * 100
    else:
        auroc = 0.0
        
    return known_acc, auroc, class_accs

# ==========================================
# Main Worker
# ==========================================
def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: 
        use_gpu = False

    if use_gpu:
        print(f"Using GPU: {options['gpu']}")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Using CPU")

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

    # Load Dataset
    print("\n" + "="*80)
    print("LOADING DDR DATASET")
    print("="*80)
    trainset = DDR(root=options['dataroot'], train=True, transform=transform_train,
                   train_class_num=options['train_class_num'], 
                   test_class_num=options['test_class_num'], 
                   includes_all_train_class=True)
    
    testset = DDR(root=options['dataroot'], train=False, transform=transform_test,
                  train_class_num=options['train_class_num'], 
                  test_class_num=options['test_class_num'], 
                  includes_all_train_class=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options['batch_size'], 
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=options['batch_size'], 
                                             shuffle=False, num_workers=0)

    options['num_classes'] = options['train_class_num']
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")
    print(f"Known classes: {options['num_classes']}")
    
    # Calculate MODERATE class weights for imbalanced dataset
    print("\nCalculating MODERATE class weights for imbalanced dataset...")
    class_counts = defaultdict(int)
    for _, label in trainset:
        if label < options['num_classes']:
            class_counts[label] += 1
    
    # Use square root of inverse frequency for more moderate weighting
    total_samples = sum(class_counts.values())
    class_weights = []
    for i in range(options['num_classes']):
        # Square root makes weights less extreme
        weight = np.sqrt(total_samples / (options['num_classes'] * class_counts[i]))
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights)
    if use_gpu:
        class_weights = class_weights.cuda()
    
    print(f"Class distribution:")
    for i in range(options['num_classes']):
        print(f"  Class {i}: {class_counts[i]} samples (weight: {class_weights[i]:.3f})")
    
    options['class_weights'] = class_weights

    # Model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    print("Architecture: ResNet50 + ARPL")
    print(f"Feature dimension: {options['feat_dim']}")
    print(f"Temperature: {options['temp']}")
    print(f"Margin loss weight: {options['weight_pl']} (INCREASED for better unknown detection)")
    
    net = ResNet50_ARPL(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
    
    # Loss
    options['use_gpu'] = use_gpu
    criterion = ARPLoss(**options)

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    # Optimizer - using different LR for backbone vs projection/criterion
    params_list = [
        {'params': net.backbone.parameters(), 'lr': options['lr'] * 0.1},  # Lower LR for pretrained backbone
        {'params': net.projection.parameters(), 'lr': options['lr']},
        {'params': criterion.parameters(), 'lr': options['lr']}
    ]
    
    optimizer = torch.optim.AdamW(params_list, weight_decay=1e-4)
    
    if options['stepsize'] > 0:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['max_epoch'])

    # Training tracking
    history = {
        'epoch': [],
        'train_loss': [],
        'known_acc': [],
        'auroc': [],
        'combined': [],
        'class_accs': []
    }
    
    start_time = time.time()
    print("\n" + "="*80)
    print(f"STARTING OPTIMIZED TRAINING FOR {options['max_epoch']} EPOCHS")
    print("Optimization Target: COMBINED METRIC (Harmonic Mean of Accuracy & AUROC)")
    print("="*80)

    best_acc = 0.0
    best_auroc = 0.0
    best_combined = 0.0
    patience = options.get('patience', 15)
    patience_counter = 0

    for epoch in range(options['max_epoch']):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{options['max_epoch']}")
        print(f"{'='*80}")
        
        # Train
        train_loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        print(f"Training Loss: {train_loss:.4f}")

        # Evaluate
        print("Evaluating...")
        known_acc, auroc, class_accs = evaluate_detailed(net, criterion, testloader, 
                                                         range(options['train_class_num']), options)
        
        # Combined metric (harmonic mean)
        if known_acc > 0 and auroc > 0:
            combined = 2 * (known_acc * auroc) / (known_acc + auroc)
        else:
            combined = 0
        
        # Log results
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['known_acc'].append(known_acc)
        history['auroc'].append(auroc)
        history['combined'].append(combined)
        history['class_accs'].append(class_accs)
        
        print(f"\nResults:")
        print(f"  Known Class Accuracy:     {known_acc:.2f}%")
        print(f"  Unknown Detection AUROC:  {auroc:.2f}%")
        print(f"  ★ Combined Score (Target): {combined:.2f}% ★")
        if class_accs:
            print(f"  Per-class accuracies:")
            for i, acc in enumerate(class_accs):
                print(f"    Class {i}: {acc:.2f}%")
        
        # Save best models
        saved = False
        if known_acc > best_acc:
            best_acc = known_acc
            save_path = os.path.join(options['outf'], 'best_acc_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'known_acc': known_acc,
                'auroc': auroc,
                'combined': combined
            }, save_path)
            print(f"  → Saved best accuracy model: {known_acc:.2f}%")
            saved = True
        
        if auroc > best_auroc:
            best_auroc = auroc
            save_path = os.path.join(options['outf'], 'best_auroc_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'known_acc': known_acc,
                'auroc': auroc,
                'combined': combined
            }, save_path)
            print(f"  → Saved best AUROC model: {auroc:.2f}%")
            saved = True
        
        if combined > best_combined:
            best_combined = combined
            patience_counter = 0
            save_path = os.path.join(options['outf'], 'best_combined_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'known_acc': known_acc,
                'auroc': auroc,
                'combined': combined
            }, save_path)
            print(f"  ★ Saved BEST COMBINED model: {combined:.2f}% ★")
            saved = True
        else:
            patience_counter += 1
        
        if not saved:
            print(f"  No improvement (patience: {patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
            break
        
        # Check if we've met target
        if known_acc >= 90.0 and auroc >= 85.0:
            print(f"\n🎯 TARGET ACHIEVED! Known Acc: {known_acc:.2f}%, AUROC: {auroc:.2f}%")
            print("Continuing training to see if we can improve further...")
        
        # Learning rate scheduling
        if options['stepsize'] > 0: 
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed}")
    print(f"\nBest Results:")
    print(f"  Best Known Accuracy: {best_acc:.2f}%")
    print(f"  Best AUROC: {best_auroc:.2f}%")
    print(f"  ★ Best Combined Score: {best_combined:.2f}% ★")
    
    # Save training history
    history_path = os.path.join(options['outf'], 'training_history.txt')
    with open(history_path, 'w') as f:
        f.write("Epoch,TrainLoss,KnownAcc,AUROC,Combined\n")
        for i in range(len(history['epoch'])):
            f.write(f"{history['epoch'][i]},{history['train_loss'][i]:.4f},"
                   f"{history['known_acc'][i]:.2f},{history['auroc'][i]:.2f},"
                   f"{history['combined'][i]:.2f}\n")
    print(f"\nTraining history saved to: {history_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Balanced DDR ARPL Training")
    
    parser.add_argument('--dataroot', type=str, default='DDR dataset')
    parser.add_argument('--outf', type=str, default='checkpoints/ddr_arpl_balanced')
    parser.add_argument('--train_class_num', type=int, default=3)
    parser.add_argument('--test_class_num', type=int, default=5)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0005)  # Slightly higher
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--stepsize', type=int, default=30)
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # ARPL specific - INCREASED margin weight for better unknown detection
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--weight_pl', type=float, default=0.3)  # INCREASED from 0.1
    parser.add_argument('--temp', type=float, default=1.0)
    
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    
    args = parser.parse_args()
    options = vars(args)
    
    if not os.path.exists(options['outf']):
        os.makedirs(options['outf'])
        
    main_worker(options)
