"""
Mahalanobis OOD with Outlier Exposure (OE) and LogitNorm
--------------------------------------------------------
Implements Steps 2 & 3 of the optimization plan:
1. Loads the best closed-set model (focal_closed_set.pth)
2. Fine-tunes with:
   - LogitNorm (stabilizes features)
   - Outlier Exposure (using Mixup and Noise as proxies)
3. Evaluates using Mahalanobis Distance
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR

# ==========================================
# 1. LogitNorm Classifier
# ==========================================
class LogitNormClassifier(nn.Module):
    def __init__(self, num_classes=3, temperature=0.1):
        super(LogitNormClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = self.backbone.fc.in_features
        
        # Replace FC with LogitNorm layer
        # Instead of standard linear layer, we normalize weights and features
        self.fc = nn.Linear(self.feature_dim, num_classes, bias=False)
        self.temperature = temperature
        
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
            
        # LogitNorm: Normalize features and weights
        # Norm of features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Norm of weights
        weights_norm = F.normalize(self.fc.weight, p=2, dim=1)
        
        # Cosine similarity
        logits = F.linear(features_norm, weights_norm)
        
        # Scale by temperature
        logits = logits / self.temperature
        
        return logits

# ==========================================
# 2. Outlier Exposure Loss
# ==========================================
def oe_loss_fn(logits_in, logits_out):
    """
    Loss = CrossEntropy(in) + 0.5 * KL(softmax(out) || uniform)
    """
    # In-distribution loss (Cross Entropy is handled in loop)
    
    # Out-of-distribution loss (KL Divergence to Uniform)
    # We want the model to be uncertain on outliers -> Uniform distribution
    
    # Softmax of outlier logits
    probs_out = F.softmax(logits_out, dim=1)
    
    # Entropy maximization (equivalent to KL to uniform)
    # Loss = -mean(sum(p * log(p))) -> Minimize negative entropy = Maximize entropy
    # Or simply: mean(sum(p * log(p)))
    
    loss_oe = torch.mean(torch.sum(probs_out * torch.log(probs_out + 1e-8), dim=1))
    
    return loss_oe

# ==========================================
# 3. Mahalanobis Utils
# ==========================================
def compute_mahalanobis_params(model, dataloader, num_classes=3, use_gpu=True):
    model.eval()
    all_features = []
    all_labels = []
    
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
    # Add tiny regularization
    reg = 1e-6 * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov + reg)
    
    return class_means, precision

def get_mahalanobis_scores(model, dataloader, class_means, precision, use_gpu=True):
    model.eval()
    scores = []
    labels_list = []
    
    with torch.no_grad():
        for data, labels in dataloader:
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True).cpu().numpy()
            
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
            
    return np.concatenate(scores), np.concatenate(labels_list)

# ==========================================
# 4. Main Training Worker
# ==========================================
def train_oe_mahalanobis(args):
    print("="*80)
    print("TRAINING: MAHALANOBIS + OUTLIER EXPOSURE + LOGITNORM")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # 1. Model Setup
    # We start with LogitNorm classifier
    model = LogitNormClassifier(num_classes=3, temperature=args['temp'])
    
    # Load pretrained weights if available (transfer learning)
    if os.path.exists(args['pretrained']):
        print(f"Loading pretrained weights from {args['pretrained']}")
        state = torch.load(args['pretrained'])
        # We need to be careful loading standard ResNet weights into LogitNorm
        # The backbone matches, but FC is different.
        # We'll load backbone only and ignore FC layers
        model_dict = model.state_dict()
        # Filter out fc layer keys
        pretrained_dict = {k: v for k, v in state.items() 
                          if 'fc' not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    
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
    
    trainset = DDR(root=args['dataroot'], train=True, transform=transform_train,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root=args['dataroot'], train=False, transform=transform_test,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args['batch_size'], shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args['batch_size'], shuffle=False)
    
    # 3. Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'])
    
    # 4. Training Loop
    print(f"\nStarting fine-tuning for {args['epochs']} epochs...")
    
    for epoch in range(args['epochs']):
        model.train()
        train_loss = 0
        
        for batch_idx, (data, labels) in enumerate(trainloader):
            if use_gpu: data, labels = data.cuda(), labels.cuda()
            
            # --- Generate Outliers (Mixup & Noise) ---
            # 1. Gaussian Noise
            noise = torch.randn_like(data)
            
            # 2. Mixup (Mix current batch with shuffled batch)
            idx = torch.randperm(data.size(0))
            data_shuffled = data[idx]
            lam = np.random.beta(1.0, 1.0)
            mixup_data = lam * data + (1 - lam) * data_shuffled
            
            # Combine outliers
            outliers = torch.cat([noise, mixup_data], dim=0)
            
            # --- Forward Pass ---
            # In-distribution
            logits_in = model(data)
            loss_ce = F.cross_entropy(logits_in, labels)
            
            # Out-of-distribution
            logits_out = model(outliers)
            loss_oe = oe_loss_fn(logits_in, logits_out)
            
            # Total Loss
            loss = loss_ce + args['lambda_oe'] * loss_oe
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx+1) % 20 == 0:
                print(f"Epoch {epoch+1} [{batch_idx+1}/{len(trainloader)}] Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, OE: {loss_oe.item():.4f})")
        
        scheduler.step()
        
        # --- Evaluation (Every 5 epochs) ---
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args['epochs']:
            print(f"\nEvaluating Epoch {epoch+1}...")
            
            # 1. Compute Mahalanobis Stats
            means, precision = compute_mahalanobis_params(model, trainloader, num_classes=3, use_gpu=use_gpu)
            
            # 2. Get Scores
            scores, labels_all = get_mahalanobis_scores(model, testloader, means, precision, use_gpu=use_gpu)
            
            # 3. Metrics
            known_mask = labels_all < 3
            unknown_mask = labels_all >= 3
            
            # AUROC
            if unknown_mask.sum() > 0:
                # Mahalanobis distance is HIGHER for unknowns
                auroc = roc_auc_score(unknown_mask, scores) * 100
                print(f"  AUROC: {auroc:.2f}%")
                
                # Known Accuracy (using Mahalanobis distance classifier)
                # Assign to class with min distance
                # We need to re-compute distances per class for this, but we only have min_dist here.
                # Let's approximate with standard accuracy for now
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, labels in testloader:
                        if use_gpu: data, labels = data.cuda(), labels.cuda()
                        outputs = model(data)
                        _, predicted = torch.max(outputs, 1)
                        mask = labels < 3
                        correct += (predicted[mask] == labels[mask]).sum().item()
                        total += mask.sum().item()
                
                acc = 100 * correct / total if total > 0 else 0
                print(f"  Known Acc: {acc:.2f}%")
                
                # Save best
                if auroc > 85:
                    torch.save(model.state_dict(), f"checkpoints/best_oe_model_ep{epoch+1}.pth")
                    print("  -> Saved best model")

if __name__ == "__main__":
    args = {
        'dataroot': 'DDR dataset',
        'pretrained': 'checkpoints/focal_closed_set.pth',
        'epochs': 30,      # Increased to 30
        'batch_size': 32,
        'lr': 0.001,
        'temp': 0.1,       # Reverted to 0.1 (Critical!)
        'lambda_oe': 0.25  # Tuned to 0.25 (Balance)
    }
    
    os.makedirs('checkpoints', exist_ok=True)
    train_oe_mahalanobis(args)
