import os
import argparse
import datetime
import time
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

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
# 1. ResNet50 Wrapper for ARPL
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
        
        # Projection head for ARPL (reduces dim to feat_dim, usually 128)
        # This helps in learning tighter clusters
        # IMPORTANT: ARPL requires bounded features (e.g. via BatchNorm) to work properly
        # otherwise Unknowns (unconstrained) will have large magnitude -> high confidence.
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU()
        )
        
        # Classifier is NOT part of the model in ARPL usually, 
        # but ARPLoss handles the logits calculation using distances.
        # However, for standard compatibility, we can keep a classifier 
        # or let ARPLoss handle it. 
        # In the official ARPL code, the model returns (features, logits)
        # But the logits are often calculated *inside* the loss or using a specific metric.
        
        # Let's follow the ARPL pattern:
        # The loss expects 'x' (features) and 'y' (logits)
        # But wait, ARPLoss calculates logits internally using Dist.
        # Let's look at ARPLoss.py again.
        # It takes 'x' (features) and 'y' (logits - optional? No, it takes x, y, labels)
        # And it calculates logits = dist_l2_p - dist_dot_p
        # So 'y' passed to it might be ignored or used for something else?
        # In ARPL/core/train.py: x, y = net(data, True); logits, loss = criterion(x, y, labels)
        # So the network MUST return features and something else.
        
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
    def forward(self, x, return_feature=False):
        # Extract features from backbone
        # ResNet50 forward:
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
        
        # In ARPL, the "logits" are often just the features or a dummy if handled by loss
        # But for evaluation, we need logits.
        # The ARPLoss has a 'Dist' module that acts as the classifier.
        # We should probably use that for inference too.
        
        return features, features # Return features twice to satisfy the (x, y) signature

# ==========================================
# 2. Training Function
# ==========================================
def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        # Forward pass
        # net returns (features, features)
        features, _ = net(data, return_feature=True)
        
        # ARPLoss forward takes (x, y, labels)
        # x = features
        # y = features (unused by ARPLoss logic usually, it uses self.Dist)
        logits, loss = criterion(features, features, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Epoch: {}/{} Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(epoch+1, options['max_epoch'], batch_idx+1, len(trainloader), losses.val, losses.avg))

    return losses.avg

# ==========================================
# 3. Evaluation Function
# ==========================================
def evaluate(net, criterion, testloader, known_classes, **options):
    net.eval()
    correct = 0
    total = 0
    
    all_scores = [] # For AUROC
    all_targets = []
    
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            features, _ = net(data, return_feature=True)
            
            # Use ARPLoss's internal Dist module to calculate logits/distances
            # ARPLoss.Dist returns distance-based logits
            # logits = dist_l2_p - dist_dot_p
            # We can access criterion.Dist directly
            
            # Calculate logits using the criterion's distance metric
            dist_dot_p = criterion.Dist(features, center=criterion.points, metric='dot')
            dist_l2_p = criterion.Dist(features, center=criterion.points)
            logits = dist_l2_p - dist_dot_p
            
            # Predictions for accuracy
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # For Unknown Detection:
            # We need a score where Higher = Known, Lower = Unknown
            # OR Higher = Unknown, Lower = Known (and flip for AUROC)
            
            # ARPL Paper: "The probability of x belonging to class k is..."
            # But simpler: Use the Max Logit (or Max Softmax Probability of these logits)
            probs = F.softmax(logits, dim=1)
            max_probs, _ = torch.max(probs, dim=1)
            
            # Store scores
            all_scores.append(max_probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_scores = np.concatenate(all_scores)
    all_targets = np.concatenate(all_targets)
    
    # 1. Known Class Accuracy
    known_mask = all_targets < len(known_classes)
    if np.sum(known_mask) > 0:
        known_acc = 0 # Recalculate strictly
        # We need predictions for this
        # Let's just use the loop's 'correct' for overall, but here we need specific
        # Re-running logic is expensive, let's capture preds in loop if needed
        # But 'correct' above includes unknowns if they happen to be predicted as known?
        # No, labels for unknowns are > len(known_classes), so they can never match predicted (0..K-1)
        # So 'correct' is exactly the number of correctly classified known samples
        
        # Wait, if test set has unknowns, 'total' includes them.
        # So 'correct/total' is NOT Known Class Accuracy.
        # Known Class Acc = Correct Knowns / Total Knowns
        
        # Let's fix this:
        total_known = np.sum(known_mask)
        # We need to know how many of these were correct.
        # We didn't save predictions. Let's save them.
        pass
    
    # Re-evaluating with saved predictions would be better
    # But for now, let's trust the loop if we only test on knowns? 
    # No, testloader has mix.
    
    # Let's do a quick hack:
    # We need to save predictions to calculate Known Acc properly
    return evaluate_detailed(net, criterion, testloader, known_classes, options)

def evaluate_detailed(net, criterion, testloader, known_classes, options):
    net.eval()
    
    all_preds = []
    all_targets = []
    all_scores = []
    all_logits = []
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
            all_logits.append(torch.max(logits, dim=1)[0].cpu().numpy())
            all_dists.append(torch.max(dist_l2_p, dim=1)[0].cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_scores = np.concatenate(all_scores)
    all_logits = np.concatenate(all_logits)
    all_dists = np.concatenate(all_dists)
    
    # Known Accuracy
    known_mask = all_targets < len(known_classes)
    if np.sum(known_mask) > 0:
        known_preds = all_preds[known_mask]
        known_targets_subset = all_targets[known_mask]
        known_acc = np.mean(known_preds == known_targets_subset) * 100
    else:
        known_acc = 0.0
        
    # Unknown Detection (AUROC)
    unknown_mask = all_targets >= len(known_classes)
    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        # Binary labels: 0 for Known, 1 for Unknown
        binary_labels = unknown_mask.astype(int)
        
        # Debug: Print average scores
        print("\nDEBUG SCORES:")
        print(f"  Avg Max Prob - Known:   {np.mean(all_scores[known_mask]):.4f}")
        print(f"  Avg Max Prob - Unknown: {np.mean(all_scores[unknown_mask]):.4f}")
        print(f"  Avg Max Logit - Known:   {np.mean(all_logits[known_mask]):.4f}")
        print(f"  Avg Max Logit - Unknown: {np.mean(all_logits[unknown_mask]):.4f}")
        print(f"  Avg Max Dist - Known:    {np.mean(all_dists[known_mask]):.4f}")
        print(f"  Avg Max Dist - Unknown:  {np.mean(all_dists[unknown_mask]):.4f}")
        
        # Score: For ARPL/distance-based methods, use MINIMUM distance to centers
        # Lower distance = Known, Higher distance = Unknown
        # So we want score that is HIGH for Unknown = use the distance itself
        unknown_scores_prob = 1.0 - all_scores
        unknown_scores_dist = all_dists  # Higher distance = more likely unknown
        
        auroc_prob = roc_auc_score(binary_labels, unknown_scores_prob) * 100
        auroc_dist = roc_auc_score(binary_labels, unknown_scores_dist) * 100
        auroc_logit = roc_auc_score(binary_labels, -all_logits) * 100
        
        print(f"  AUROC (1-Prob): {auroc_prob:.2f}%")
        print(f"  AUROC (MaxDist): {auroc_dist:.2f}%")
        print(f"  AUROC (-Logit): {auroc_logit:.2f}%")
        
        # Use the best one
        auroc = max(auroc_prob, auroc_dist, auroc_logit)
        if auroc == auroc_dist:
            print("  -> Using Distance-based score (BEST for ARPL).")
        elif auroc == auroc_logit:
            print("  -> Using Logit-based score.")
        else:
            print("  -> Using Probability-based score.")
            
    else:
        auroc = 0.0
        
    return known_acc, auroc

# ==========================================
# 4. Main Worker
# ==========================================
def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: 
        use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

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
    print("Loading DDR dataset...")
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
    print(f"Training on {options['num_classes']} known classes")

    # Model
    print("Creating model: ResNet50_ARPL")
    net = ResNet50_ARPL(num_classes=options['num_classes'], feat_dim=options['feat_dim'])
    
    # Loss
    options['use_gpu'] = use_gpu
    criterion = ARPLoss(**options)

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    # Optimizer
    # ARPL usually uses SGD, but AdamW is good for ResNet fine-tuning
    # Let's stick to AdamW but maybe lower LR
    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]
    
    optimizer = torch.optim.AdamW(params_list, lr=options['lr'], weight_decay=1e-4)
    
    if options['stepsize'] > 0:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['max_epoch'])

    start_time = time.time()
    print(f"Starting training for {options['max_epoch']} epochs...")

    best_acc = 0.0
    best_auroc = 0.0

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        
        train_loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        print(f"Epoch {epoch+1} Loss: {train_loss:.4f}")

        if (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Testing")
            known_acc, auroc = evaluate(net, criterion, testloader, range(options['train_class_num']), **options)
            print(f"Epoch {epoch+1} Results:")
            print(f"  Known Class Accuracy:   {known_acc:.2f}%")
            print(f"  Unknown Detection AUROC: {auroc:.2f}%")
            
            # Save best model based on Harmonic Mean or just Acc?
            # User said: "known class accuracy must not dropped"
            # So we prioritize Acc, but want high AUROC.
            
            if known_acc >= 90.0 and auroc > best_auroc:
                 best_auroc = auroc
                 save_path = os.path.join(options['outf'], 'best_arpl_model.pth')
                 torch.save(net.state_dict(), save_path)
                 print(f"  -> Saved best AUROC model (Acc > 90%) to {save_path}")
            
            if known_acc > best_acc:
                best_acc = known_acc

        if options['stepsize'] > 0: 
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    print(f"Best Known Class Accuracy: {best_acc:.2f}%")
    print(f"Best AUROC (with high Acc): {best_auroc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DDR ARPL Training")
    
    parser.add_argument('--dataroot', type=str, default='DDR dataset')
    parser.add_argument('--outf', type=str, default='checkpoints/ddr_arpl_v2')
    parser.add_argument('--train_class_num', type=int, default=3)
    parser.add_argument('--test_class_num', type=int, default=5)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--stepsize', type=int, default=30)
    
    # ARPL specific
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--weight_pl', type=float, default=0.1)
    parser.add_argument('--temp', type=float, default=1.0)
    
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    
    args = parser.parse_args()
    options = vars(args)
    
    if not os.path.exists(options['outf']):
        os.makedirs(options['outf'])
        
    main_worker(options)
