import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'OSR', 'ARPL'))

from datasets import DDR
from OSR.ARPL.loss.ARPLoss import ARPLoss
from OSR.ARPL.utils import AverageMeter

# ==========================================
# Model Definition (Same as training)
# ==========================================
class ResNet50_ARPL(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128, use_gpu=True):
        super(ResNet50_ARPL, self).__init__()
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        
    def forward(self, x, return_feature=False):
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
        
        features = self.projection(x)
        features = F.normalize(features, p=2, dim=1)
        
        return features, features

# ==========================================
# Baseline: Simple ResNet50 with CrossEntropy
# ==========================================
class SimpleResNet50(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleResNet50, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# ==========================================
# Diagnostic Evaluation
# ==========================================
def evaluate_quick(net, criterion, testloader, num_classes, arpl_mode=True, use_gpu=True):
    net.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            
            if arpl_mode:
                features, _ = net(data, return_feature=True)
                dist_dot_p = criterion.Dist(features, center=criterion.points, metric='dot')
                dist_l2_p = criterion.Dist(features, center=criterion.points)
                logits = dist_l2_p - dist_dot_p
            else:
                logits = net(data)
            
            _, predicted = torch.max(logits, 1)
            
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Known samples only
    known_mask = all_targets < num_classes
    if np.sum(known_mask) > 0:
        known_preds = all_preds[known_mask]
        known_targets = all_targets[known_mask]
        known_acc = np.mean(known_preds == known_targets) * 100
        
        # Per-class
        print(f"  Overall Known Accuracy: {known_acc:.2f}%")
        for c in range(num_classes):
            class_mask = known_targets == c
            if np.sum(class_mask) > 0:
                class_acc = np.mean(known_preds[class_mask] == c) * 100
                print(f"    Class {c}: {class_acc:.2f}% ({np.sum(class_mask)} samples)")
        
        return known_acc
    return 0.0

# ==========================================
# Diagnostic Training
# ==========================================
def diagnostic_train(net, criterion, optimizer, trainloader, num_classes, arpl_mode=True, use_gpu=True):
    net.train()
    
    losses = AverageMeter()
    ce_losses = AverageMeter()
    margin_losses = AverageMeter()
    
    batch_count = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        
        if arpl_mode:
            features, _ = net(data, return_feature=True)
            logits, loss = criterion(features, features, labels)
            
            # Manual calculation for debugging
            dist_dot_p = criterion.Dist(features, center=criterion.points, metric='dot')
            dist_l2_p = criterion.Dist(features, center=criterion.points)
            logits_manual = dist_l2_p - dist_dot_p
            
            ce_loss = F.cross_entropy(logits_manual / criterion.temp, labels)
            center_batch = criterion.points[labels, :]
            _dis_known = (features - center_batch).pow(2).mean(1)
            target = torch.ones(_dis_known.size()).cuda() if use_gpu else torch.ones(_dis_known.size())
            margin_loss = criterion.margin_loss(criterion.radius, _dis_known, target)
            
            ce_losses.update(ce_loss.item(), labels.size(0))
            margin_losses.update(margin_loss.item(), labels.size(0))
            
            # Debug info for first batch
            if batch_idx == 0:
                print(f"\n  [Batch 0 Debug]")
                print(f"    Features shape: {features.shape}, mean: {features.mean().item():.4f}, std: {features.std().item():.4f}")
                print(f"    Features norm mean: {torch.norm(features, dim=1).mean().item():.4f}")
                print(f"    Dist L2 range: [{dist_l2_p.min().item():.4f}, {dist_l2_p.max().item():.4f}]")
                print(f"    Dist Dot range: [{dist_dot_p.min().item():.4f}, {dist_dot_p.max().item():.4f}]")
                print(f"    Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
                print(f"    CE Loss: {ce_loss.item():.4f}")
                print(f"    Margin Loss: {margin_loss.item():.4f}")
                print(f"    Total Loss: {loss.item():.4f}")
                print(f"    Radius: {criterion.radius.item():.4f}")
                print(f"    Center norms: {torch.norm(criterion.points, dim=1).mean().item():.4f}")
        else:
            logits = net(data)
            loss = F.cross_entropy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), labels.size(0))
        
        batch_count += 1
        if batch_count >= 50:  # Only train on first 50 batches for diagnosis
            break
    
    if arpl_mode:
        print(f"  Avg Loss: {losses.avg:.4f} (CE: {ce_losses.avg:.4f}, Margin: {margin_losses.avg:.4f})")
    else:
        print(f"  Avg Loss: {losses.avg:.4f}")
    
    return losses.avg

# ==========================================
# Main Diagnostic
# ==========================================
def main():
    print("="*80)
    print("ARPL TRAINING DIAGNOSTIC")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    print(f"\nUsing GPU: {use_gpu}")
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("\nLoading DDR dataset...")
    trainset = DDR(root='DDR dataset', train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    testset = DDR(root='DDR dataset', train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train set: {len(trainset)} samples")
    print(f"Test set: {len(testset)} samples")
    
    # ==========================================
    # TEST 1: Baseline - Simple ResNet50
    # ==========================================
    print("\n" + "="*80)
    print("TEST 1: BASELINE - Simple ResNet50 with CrossEntropy")
    print("="*80)
    
    print("\nCreating baseline model...")
    baseline_net = SimpleResNet50(num_classes=3)
    if use_gpu:
        baseline_net = baseline_net.cuda()
    
    baseline_optimizer = torch.optim.AdamW(baseline_net.parameters(), lr=0.0001, weight_decay=1e-4)
    
    print("\nTraining baseline for 2 epochs...")
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2:")
        diagnostic_train(baseline_net, None, baseline_optimizer, trainloader, 3, 
                        arpl_mode=False, use_gpu=use_gpu)
        
        print("  Evaluating...")
        baseline_acc = evaluate_quick(baseline_net, None, testloader, 3, 
                                     arpl_mode=False, use_gpu=use_gpu)
    
    print(f"\n✓ Baseline Final Accuracy: {baseline_acc:.2f}%")
    
    # ==========================================
    # TEST 2: ARPL with current configuration
    # ==========================================
    print("\n" + "="*80)
    print("TEST 2: ARPL with Current Configuration")
    print("="*80)
    
    print("\nCreating ARPL model...")
    arpl_net = ResNet50_ARPL(num_classes=3, feat_dim=128)
    
    options = {
        'use_gpu': use_gpu,
        'num_classes': 3,
        'feat_dim': 128,
        'weight_pl': 0.1,
        'temp': 1.0
    }
    arpl_criterion = ARPLoss(**options)
    
    if use_gpu:
        arpl_net = arpl_net.cuda()
        arpl_criterion = arpl_criterion.cuda()
    
    params_list = [{'params': arpl_net.parameters()},
                   {'params': arpl_criterion.parameters()}]
    arpl_optimizer = torch.optim.AdamW(params_list, lr=0.0001, weight_decay=1e-4)
    
    print("\nInitial center statistics:")
    print(f"  Centers shape: {arpl_criterion.points.shape}")
    print(f"  Centers norm mean: {torch.norm(arpl_criterion.points, dim=1).mean().item():.4f}")
    print(f"  Centers mean: {arpl_criterion.points.mean().item():.4f}")
    print(f"  Centers std: {arpl_criterion.points.std().item():.4f}")
    
    print("\nTraining ARPL for 2 epochs...")
    for epoch in range(2):
        print(f"\nEpoch {epoch+1}/2:")
        diagnostic_train(arpl_net, arpl_criterion, arpl_optimizer, trainloader, 3,
                        arpl_mode=True, use_gpu=use_gpu)
        
        print("  Evaluating...")
        arpl_acc = evaluate_quick(arpl_net, arpl_criterion, testloader, 3,
                                 arpl_mode=True, use_gpu=use_gpu)
        
        print(f"\n  Center statistics after epoch {epoch+1}:")
        print(f"    Centers norm mean: {torch.norm(arpl_criterion.points, dim=1).mean().item():.4f}")
        print(f"    Centers mean: {arpl_criterion.points.mean().item():.4f}")
        print(f"    Centers std: {arpl_criterion.points.std().item():.4f}")
        print(f"    Radius: {arpl_criterion.radius.item():.4f}")
    
    print(f"\n✓ ARPL Final Accuracy: {arpl_acc:.2f}%")
    
    # ==========================================
    # TEST 3: ARPL with different hyperparameters
    # ==========================================
    print("\n" + "="*80)
    print("TEST 3: ARPL with Modified Hyperparameters")
    print("="*80)
    
    print("\nTesting different configurations...")
    
    configs = [
        {'temp': 0.1, 'weight_pl': 0.01, 'lr': 0.001, 'name': 'Lower temp, lower weight_pl, higher LR'},
        {'temp': 1.0, 'weight_pl': 0.0, 'lr': 0.0001, 'name': 'No margin loss (pure CE)'},
        {'temp': 0.5, 'weight_pl': 0.1, 'lr': 0.0005, 'name': 'Medium temp, higher LR'},
    ]
    
    best_config = None
    best_acc = 0.0
    
    for config in configs:
        print(f"\n--- Config: {config['name']} ---")
        print(f"    temp={config['temp']}, weight_pl={config['weight_pl']}, lr={config['lr']}")
        
        test_net = ResNet50_ARPL(num_classes=3, feat_dim=128)
        test_options = {
            'use_gpu': use_gpu,
            'num_classes': 3,
            'feat_dim': 128,
            'weight_pl': config['weight_pl'],
            'temp': config['temp']
        }
        test_criterion = ARPLoss(**test_options)
        
        if use_gpu:
            test_net = test_net.cuda()
            test_criterion = test_criterion.cuda()
        
        params_list = [{'params': test_net.parameters()},
                      {'params': test_criterion.parameters()}]
        test_optimizer = torch.optim.AdamW(params_list, lr=config['lr'], weight_decay=1e-4)
        
        # Train for 2 epochs
        for epoch in range(2):
            diagnostic_train(test_net, test_criterion, test_optimizer, trainloader, 3,
                           arpl_mode=True, use_gpu=use_gpu)
        
        # Evaluate
        test_acc = evaluate_quick(test_net, test_criterion, testloader, 3,
                                 arpl_mode=True, use_gpu=use_gpu)
        
        print(f"    Result: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_config = config
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print(f"\n1. Baseline ResNet50 (2 epochs): {baseline_acc:.2f}%")
    print(f"2. ARPL Current Config (2 epochs): {arpl_acc:.2f}%")
    print(f"3. ARPL Best Config (2 epochs): {best_acc:.2f}%")
    if best_config:
        print(f"   Best config: {best_config['name']}")
        print(f"   Parameters: temp={best_config['temp']}, weight_pl={best_config['weight_pl']}, lr={best_config['lr']}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    if baseline_acc < 50:
        print("⚠️  WARNING: Even baseline model is struggling (<50% acc)")
        print("   → Problem might be with data, transforms, or learning rate")
    elif baseline_acc >= 70:
        print("✓ Baseline model works well (>70% acc)")
    
    if arpl_acc < baseline_acc * 0.5:
        print("⚠️  CRITICAL: ARPL performing much worse than baseline")
        print("   → Likely issue with ARPL loss calculation or distance metrics")
    elif arpl_acc < baseline_acc * 0.8:
        print("⚠️  ARPL underperforming baseline significantly")
        print("   → Hyperparameters may need tuning")
    else:
        print("✓ ARPL comparable to baseline")
    
    if best_acc > arpl_acc * 1.2:
        print(f"✓ Found better hyperparameters (+{(best_acc - arpl_acc):.1f}%)")
        print("   → Recommend using best config for full training")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
