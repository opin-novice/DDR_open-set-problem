import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.covariance import EmpiricalCovariance
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_one_seed(seed, output_dir):
    print(f"\n>>> TRAINING SEED {seed} <<<")
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    # Data
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    trainset = DDR(root='DDR dataset', train=True, transform=transform_train, 
                   train_class_num=5, test_class_num=5, includes_all_train_class=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
    
    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Train Loop (30 Epochs)
    for epoch in range(30):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"  Seed {seed} | Epoch {epoch+1}/30 complete")
        
    # Save Model
    save_path = os.path.join(output_dir, f'model_seed_{seed}.pth')
    torch.save(model.state_dict(), save_path)
    return save_path

def evaluate_one_seed(model_path, seed):
    print(f"\n>>> EVALUATING SEED {seed} <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Datasets
    ddr_train = DDR(root='DDR dataset', train=True, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_test = DDR(root='DDR dataset', train=False, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
    acrima = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
    
    loaders = {
        'train': DataLoader(ddr_train, batch_size=32, shuffle=False, num_workers=4),
        'test_known': DataLoader(ddr_test, batch_size=32, shuffle=False, num_workers=4),
        'test_unknown': DataLoader(acrima, batch_size=32, shuffle=False, num_workers=4)
    }
    
    # Extract Features Helper
    def get_feats(loader):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                # Forward pass to penultimate
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                f = torch.flatten(x, 1)
                feats.append(f.cpu().numpy())
                labels.append(y.numpy())
        return np.concatenate(feats), np.concatenate(labels)

    train_feats, train_labels = get_feats(loaders['train'])
    known_feats, known_labels = get_feats(loaders['test_known'])
    unknown_feats, _ = get_feats(loaders['test_unknown'])
    
    # Mahalanobis Stats
    class_means = []
    centered_data = []
    for c in range(5):
        mask = train_labels == c
        c_feats = train_feats[mask]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_data.append(c_feats - mean)
        
    cov = EmpiricalCovariance().fit(np.concatenate(centered_data))
    precision = cov.precision_
    
    # Scores
    def get_scores(feats):
        dists = []
        for c in range(5):
            centered = feats - class_means[c]
            d = np.sqrt(np.sum(centered @ precision * centered, axis=1))
            dists.append(d)
        return np.min(dists, axis=0), np.argmin(dists, axis=0)
        
    scores_known, preds_known = get_scores(known_feats)
    scores_unknown, _ = get_scores(unknown_feats)
    
    # Metrics
    known_acc = 100.0 * (preds_known == known_labels).sum() / len(known_labels)
    
    y_true = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
    y_scores = np.concatenate([scores_known, scores_unknown])
    auroc = roc_auc_score(y_true, y_scores) * 100
    
    h_mean = 2 * (known_acc * auroc) / (known_acc + auroc)
    
    # Class-wise Acc
    cm = confusion_matrix(known_labels, preds_known)
    class_accs = cm.diagonal() / cm.sum(axis=1) * 100
    
    return {
        'Seed': seed,
        'Known Acc': known_acc,
        'AUROC': auroc,
        'H-mean': h_mean,
        'No_DR': class_accs[0],
        'Mild': class_accs[1],
        'Moderate': class_accs[2],
        'Severe': class_accs[3],
        'Proliferative': class_accs[4]
    }

def main():
    SEEDS = [42, 1, 2024]
    results = []
    output_dir = 'reproducibility_cross_dataset'
    
    print("="*80)
    print("STARTING 3-SEED CROSS-DATASET REPRODUCIBILITY RUN")
    print("="*80)
    
    for seed in SEEDS:
        model_path = train_one_seed(seed, output_dir)
        metrics = evaluate_one_seed(model_path, seed)
        results.append(metrics)
        print(f"Seed {seed} Results: Acc={metrics['Known Acc']:.2f}, AUROC={metrics['AUROC']:.2f}")
        
    # Aggregate
    df = pd.DataFrame(results)
    mean = df.mean()
    std = df.std()
    
    # Generate Report
    report = f"""# Cross-Dataset OSR Reproducibility Report

## Experiment Setup
- **Task:** Open Set Recognition (DDR vs ACRIMA)
- **Known Classes:** 5 (No_DR, Mild, Moderate, Severe, Proliferative)
- **Unknown Class:** Glaucoma (ACRIMA Dataset)
- **Model:** ResNet50 (Pretrained ImageNet)
- **Method:** Mahalanobis Distance
- **Seeds:** {SEEDS}

## Aggregated Results (Mean ± Std)
| Metric | Mean | Std |
| :--- | :--- | :--- |
| **Known Accuracy** | **{mean['Known Acc']:.2f}%** | ±{std['Known Acc']:.2f} |
| **AUROC** | **{mean['AUROC']:.2f}%** | ±{std['AUROC']:.2f} |
| **H-mean** | **{mean['H-mean']:.2f}%** | ±{std['H-mean']:.2f} |

## Detailed Results
{df.to_markdown(index=False, floatfmt=".2f")}

## Class-wise Accuracy (Known)
| Class | Mean Acc | Std |
| :--- | :--- | :--- |
| No_DR | {mean['No_DR']:.2f}% | ±{std['No_DR']:.2f} |
| Mild | {mean['Mild']:.2f}% | ±{std['Mild']:.2f} |
| Moderate | {mean['Moderate']:.2f}% | ±{std['Moderate']:.2f} |
| Severe | {mean['Severe']:.2f}% | ±{std['Severe']:.2f} |
| Proliferative | {mean['Proliferative']:.2f}% | ±{std['Proliferative']:.2f} |
"""
    
    with open('reproducibility.md', 'w') as f:
        f.write(report)
        
    print("\n" + "="*80)
    print("REPRODUCIBILITY RUN COMPLETE")
    print("Report saved to reproducibility.md")
    print("="*80)

if __name__ == "__main__":
    main()
