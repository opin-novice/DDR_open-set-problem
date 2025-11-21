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

# Model definition (same as training)
class ResNet50_ARPL(nn.Module):
    def __init__(self, num_classes=3, feat_dim=128, use_gpu=True):
        super(ResNet50_ARPL, self).__init__()
        
        self.backbone = models.resnet50(weights=None)
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

def evaluate_checkpoint(checkpoint_path, checkpoint_name):
    print(f"\n{'='*80}")
    print(f"EVALUATING: {checkpoint_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
    
    # Display checkpoint info
    checkpoint_info = os.stat(checkpoint_path)
    print(f"Checkpoint size: {checkpoint_info.st_size / (1024*1024):.2f} MB")
    print(f"Last modified: {checkpoint_info.st_mtime}")
    import datetime
    print(f"Last modified (readable): {datetime.datetime.fromtimestamp(checkpoint_info.st_mtime)}")
    
    # Setup
    use_gpu = torch.cuda.is_available()
    print(f"\nUsing GPU: {use_gpu}")
    
    # Transforms
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load test dataset
    print("\nLoading DDR test dataset...")
    testset = DDR(root='DDR dataset', train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, 
                  includes_all_train_class=True)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, 
                                             shuffle=False, num_workers=0)
    
    print(f"Test set size: {len(testset)} samples")
    
    # Create model
    print("\nCreating model...")
    net = ResNet50_ARPL(num_classes=3, feat_dim=128)
    
    # Load checkpoint
    print(f"Loading checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(state_dict)
    
    # Create criterion for evaluation
    options = {
        'use_gpu': use_gpu,
        'num_classes': 3,
        'feat_dim': 128,
        'weight_pl': 0.1,
        'temp': 1.0
    }
    criterion = ARPLoss(**options)
    
    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()
    
    # Evaluate
    print("\n" + "="*80)
    print("RUNNING EVALUATION...")
    print("="*80 + "\n")
    
    net.eval()
    
    all_preds = []
    all_targets = []
    all_scores = []
    all_logits = []
    all_dists = []
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(testloader):
            if use_gpu:
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
            all_logits.append(torch.max(logits, dim=1)[0].cpu().numpy())
            all_dists.append(torch.min(dist_l2_p, dim=1)[0].cpu().numpy())  # MIN dist to closest center
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * 16}/{len(testset)} samples...")
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)
    all_logits = np.concatenate(all_logits)
    all_dists = np.concatenate(all_dists)
    
    # Calculate metrics
    known_mask = all_targets < 3
    unknown_mask = all_targets >= 3
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    # Dataset statistics
    print("Dataset Statistics:")
    print(f"  Total samples: {len(all_targets)}")
    print(f"  Known samples (classes 0, 1, 2): {np.sum(known_mask)}")
    print(f"  Unknown samples (classes 3, 4): {np.sum(unknown_mask)}")
    
    # Known Class Accuracy
    if np.sum(known_mask) > 0:
        known_preds = all_preds[known_mask]
        known_targets_subset = all_targets[known_mask]
        known_acc = np.mean(known_preds == known_targets_subset) * 100
        print(f"\n✓ Known Class Accuracy: {known_acc:.2f}%")
        
        # Per-class accuracy
        print("\n  Per-class accuracy:")
        for c in range(3):
            class_mask = known_targets_subset == c
            if np.sum(class_mask) > 0:
                class_acc = np.mean(known_preds[class_mask] == c) * 100
                class_count = np.sum(class_mask)
                print(f"    Class {c}: {class_acc:.2f}% ({class_count} samples)")
    else:
        known_acc = 0.0
        print("No known samples in test set!")
    
    # Unknown Detection (AUROC)
    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        binary_labels = unknown_mask.astype(int)
        
        print("\n" + "-"*80)
        print("Score Analysis:")
        print(f"  Average Max Probability - Known:   {np.mean(all_scores[known_mask]):.4f}")
        print(f"  Average Max Probability - Unknown: {np.mean(all_scores[unknown_mask]):.4f}")
        print(f"  Average Max Logit - Known:   {np.mean(all_logits[known_mask]):.4f}")
        print(f"  Average Max Logit - Unknown: {np.mean(all_logits[unknown_mask]):.4f}")
        print(f"  Average Min Distance - Known:    {np.mean(all_dists[known_mask]):.4f}")
        print(f"  Average Min Distance - Unknown:  {np.mean(all_dists[unknown_mask]):.4f}")
        
        # Calculate AUROC with different scoring methods
        unknown_scores_prob = 1.0 - all_scores
        unknown_scores_dist = all_dists  # Higher distance = more likely unknown
        unknown_scores_logit = -all_logits
        
        auroc_prob = roc_auc_score(binary_labels, unknown_scores_prob) * 100
        auroc_dist = roc_auc_score(binary_labels, unknown_scores_dist) * 100
        auroc_logit = roc_auc_score(binary_labels, unknown_scores_logit) * 100
        
        print("\n" + "-"*80)
        print("Unknown Detection AUROC:")
        print(f"  Using (1 - Max Probability): {auroc_prob:.2f}%")
        print(f"  Using Min Distance to Centers: {auroc_dist:.2f}%")
        print(f"  Using (-Max Logit): {auroc_logit:.2f}%")
        
        best_auroc = max(auroc_prob, auroc_dist, auroc_logit)
        if best_auroc == auroc_dist:
            method = "Distance-based (typical for ARPL)"
        elif best_auroc == auroc_logit:
            method = "Logit-based"
        else:
            method = "Probability-based"
        
        print(f"\n✓ Best Unknown Detection AUROC: {best_auroc:.2f}% ({method})")
    else:
        best_auroc = 0.0
        print("\nNo unknown samples or no known samples in test set!")
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Known Class Accuracy:     {known_acc:.2f}%")
    print(f"Unknown Detection AUROC:  {best_auroc:.2f}%")
    print(f"Harmonic Mean (F1-like):  {2 * (known_acc * best_auroc) / (known_acc + best_auroc) if (known_acc + best_auroc) > 0 else 0:.2f}%")
    print("="*80 + "\n")
    
    return known_acc, best_auroc

if __name__ == '__main__':
    # Check all available checkpoints
    checkpoints_to_check = [
        ('checkpoints/ddr_arpl_v2/best_arpl_model.pth', 'DDR ARPL V2 (MOST RECENT)'),
        ('checkpoints/ddr_resnet/best_resnet50_model.pth', 'DDR ResNet50'),
        ('checkpoints/ddr_arpl/best_arpl_resnet50.pth', 'DDR ARPL ResNet50'),
    ]
    
    print("\n" + "="*80)
    print("LAST TRAINING RESULTS - POST POWER OUTAGE CHECK")
    print("="*80)
    
    # Evaluate the most recent checkpoint
    evaluate_checkpoint(checkpoints_to_check[0][0], checkpoints_to_check[0][1])
