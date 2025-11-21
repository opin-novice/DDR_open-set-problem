"""
Save Features and Statistics for Baseline Model
-----------------------------------------------
Extracts penultimate features (2048-dim) from the trained model.
Saves:
- Train/Test Features and Labels (.npy)
- Class Means and Precision Matrix (.npy)

Useful for visualization (t-SNE) and future OOD analysis.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.covariance import EmpiricalCovariance

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from datasets import DDR

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
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

def extract_features(model, dataloader, use_gpu=True):
    model.eval()
    all_features = []
    all_labels = []
    print("  Extracting...", end="", flush=True)
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            if use_gpu: data = data.cuda()
            features = model(data, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            if (i+1) % 50 == 0: print(".", end="", flush=True)
    print(" Done.")
    return np.concatenate(all_features), np.concatenate(all_labels)

def compute_stats(features, labels, num_classes=3):
    class_means = []
    centered_features = []
    for c in range(num_classes):
        mask = labels == c
        c_feats = features[mask]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_features.append(c_feats - mean)
    
    all_centered = np.concatenate(centered_features, axis=0)
    cov = EmpiricalCovariance().fit(all_centered).covariance_
    reg = 0.01 * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov + reg)
    return np.array(class_means), precision

def main():
    print("="*80)
    print("SAVING BASELINE FEATURES & STATS")
    print("="*80)
    
    use_gpu = torch.cuda.is_available()
    
    # 1. Load Model
    model = ResNet50Classifier(num_classes=3)
    ckpt_path = 'checkpoints/baseline_focal_mahalanobis.pth'
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found.")
        return
        
    model.load_state_dict(torch.load(ckpt_path))
    if use_gpu: model = model.cuda()
    print(f"Loaded model from {ckpt_path}")
    
    # 2. Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    trainset = DDR(root='DDR dataset', train=True, transform=transform,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root='DDR dataset', train=False, transform=transform,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)
    
    # 3. Extract
    print("\nExtracting Training Features...")
    train_X, train_y = extract_features(model, trainloader, use_gpu)
    
    print("\nExtracting Test Features...")
    test_X, test_y = extract_features(model, testloader, use_gpu)
    
    # 4. Compute Stats (on known train data)
    print("\nComputing Mahalanobis Statistics...")
    known_mask = train_y < 3
    means, precision = compute_stats(train_X[known_mask], train_y[known_mask])
    
    # 5. Save
    save_dir = 'saved_features_baseline'
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, 'train_features.npy'), train_X)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_y)
    np.save(os.path.join(save_dir, 'test_features.npy'), test_X)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_y)
    np.save(os.path.join(save_dir, 'class_means.npy'), means)
    np.save(os.path.join(save_dir, 'precision.npy'), precision)
    
    print(f"\nSUCCESS! All artifacts saved to '{save_dir}/'")
    print(f"  - train_features.npy: {train_X.shape}")
    print(f"  - test_features.npy:  {test_X.shape}")
    print(f"  - class_means.npy:    {means.shape}")
    print(f"  - precision.npy:      {precision.shape}")

if __name__ == "__main__":
    main()
