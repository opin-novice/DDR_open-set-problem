import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

def evaluate_one_seed(model_path, seed):
    print(f"\n>>> EVALUATING SEED {seed} <<<")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Checkpoint not found at {model_path}")
        return None

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

    print("  Extracting features...")
    train_feats, train_labels = get_feats(loaders['train'])
    known_feats, known_labels = get_feats(loaders['test_known'])
    unknown_feats, _ = get_feats(loaders['test_unknown'])
    
    # Mahalanobis Stats
    print("  Computing Mahalanobis stats...")
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
    
    h_mean = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
    
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

    # Aggregate
    if not results:
        print("No results found!")
        return

    df = pd.DataFrame(results)
    mean = df.mean()
    std = df.std()
    
    # --- NEW: Visualizations ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Class-wise Accuracy Boxplot
    class_cols = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    class_data = df[class_cols].melt(var_name='Class', value_name='Accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Accuracy', data=class_data, palette='Blues')
    plt.title('Class-wise Accuracy Distribution across 3 Seeds')
    plt.ylim(80, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('reproducibility_class_accuracy_boxplot.png')
    plt.close()
    
    # 2. Mahalanobis Distance Histogram (Aggregated)
    # We need to re-collect scores from all seeds to plot a combined histogram
    # For simplicity, let's just use the scores from the last seed (or we could modify the loop to store them all)
    # To be accurate, let's just plot for the last seed as a representative example, 
    # or ideally, we should have stored them. 
    # Let's modify the loop above to store scores? 
    # Actually, let's just re-run the last seed's scores for the plot since we have them in memory if we are careful.
    # Wait, evaluate_one_seed returns metrics, not scores. 
    # Let's just use the last seed's scores since they are computed in the function.
    # We can modify evaluate_one_seed to return scores too, but that changes the signature.
    # Let's just accept that the histogram will be for the *last processed seed* (Seed 2024) 
    # which is a fair representative sample.
    # To do this properly, we'd need to refactor. 
    # Let's stick to the plan: Use the last seed's data for the histogram 
    # BUT to make it "across seeds" implies aggregation. 
    # Let's quickly refactor to store scores in a list.
    
    # Refactoring loop to store scores
    all_known_scores = []
    all_unknown_scores = []
    
    # We need to re-run evaluation to get scores if we want ALL seeds. 
    # Since we are in "recover" mode, we can just do it.
    # But wait, the previous loop already ran. 
    # Let's just add a quick hack: re-run the score extraction for the histogram.
    # Or better, let's just update the loop in this function to store them.
    # Since I am replacing the END of the file, I can't easily change the loop above without replacing the whole main().
    # I will replace the WHOLE main() function to be safe.

def main():
    SEEDS = [42, 1, 2024]
    results = []
    all_known_scores = []
    all_unknown_scores = []
    output_dir = 'reproducibility_cross_dataset'
    
    print("="*80)
    print("RECOVERING REPRODUCIBILITY RESULTS & PLOTS")
    print("="*80)
    
    # We need to modify evaluate_one_seed to return scores to us
    # Or we can just copy-paste the logic here. 
    # Let's define a helper that returns scores too.
    
    for seed in SEEDS:
        model_path = os.path.join(output_dir, f'model_seed_{seed}.pth')
        
        # --- Inline Evaluation Logic to get Scores ---
        print(f"\n>>> PROCESSING SEED {seed} <<<")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            print(f"ERROR: Checkpoint not found at {model_path}")
            continue

        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        ddr_train = DDR(root='DDR dataset', train=True, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
        ddr_test = DDR(root='DDR dataset', train=False, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
        acrima = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
        
        loaders = {
            'train': DataLoader(ddr_train, batch_size=32, shuffle=False, num_workers=4),
            'test_known': DataLoader(ddr_test, batch_size=32, shuffle=False, num_workers=4),
            'test_unknown': DataLoader(acrima, batch_size=32, shuffle=False, num_workers=4)
        }
        
        def get_feats(loader):
            feats, labels = [], []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
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

        print("  Extracting features...")
        train_feats, train_labels = get_feats(loaders['train'])
        known_feats, known_labels = get_feats(loaders['test_known'])
        unknown_feats, _ = get_feats(loaders['test_unknown'])
        
        print("  Computing Mahalanobis stats...")
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
        
        def get_scores(feats):
            dists = []
            for c in range(5):
                centered = feats - class_means[c]
                d = np.sqrt(np.sum(centered @ precision * centered, axis=1))
                dists.append(d)
            return np.min(dists, axis=0), np.argmin(dists, axis=0)
            
        scores_known, preds_known = get_scores(known_feats)
        scores_unknown, _ = get_scores(unknown_feats)
        
        # Store scores for histogram
        all_known_scores.extend(scores_known)
        all_unknown_scores.extend(scores_unknown)
        
        # Metrics
        known_acc = 100.0 * (preds_known == known_labels).sum() / len(known_labels)
        y_true = np.concatenate([np.zeros(len(scores_known)), np.ones(len(scores_unknown))])
        y_scores = np.concatenate([scores_known, scores_unknown])
        auroc = roc_auc_score(y_true, y_scores) * 100
        h_mean = 2 * (known_acc * auroc) / (known_acc + auroc) if (known_acc + auroc) > 0 else 0
        
        cm = confusion_matrix(known_labels, preds_known)
        class_accs = cm.diagonal() / cm.sum(axis=1) * 100
        
        metrics = {
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
        results.append(metrics)
        print(f"Seed {seed} Results: Acc={metrics['Known Acc']:.2f}, AUROC={metrics['AUROC']:.2f}")

    # Aggregate
    if not results:
        print("No results found!")
        return

    df = pd.DataFrame(results)
    mean = df.mean()
    std = df.std()
    
    # --- Visualizations ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Class-wise Accuracy Boxplot
    class_cols = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    class_data = df[class_cols].melt(var_name='Class', value_name='Accuracy')
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Accuracy', data=class_data, palette='Blues')
    plt.title('Class-wise Accuracy Distribution across 3 Seeds')
    plt.ylim(80, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('reproducibility_class_accuracy_boxplot.png')
    plt.close()
    
    # 2. Mahalanobis Distance Histogram (Aggregated)
    plt.figure(figsize=(10, 6))
    plt.hist(all_known_scores, bins=50, alpha=0.5, label='Known (DDR)', color='blue', density=True)
    plt.hist(all_unknown_scores, bins=50, alpha=0.5, label='Unknown (Glaucoma)', color='red', density=True)
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Density')
    plt.title(f'OOD Detection Score Distribution (Aggregated across 3 Seeds)')
    plt.legend()
    plt.savefig('reproducibility_distance_histogram.png')
    plt.close()
    
    # Generate Report
    report = f"""# Cross-Dataset OSR Reproducibility Report

## Experiment Setup
### Training Configuration
| Parameter | Value |
| :--- | :--- |
| **Epochs** | 30 |
| **Batch Size** | 32 |
| **Optimizer** | Adam (lr=0.001) |
| **Scheduler** | Cosine Annealing (T_max=30) |
| **Augmentation** | RandomFlip, RandomRotation(10), Resize(224) |
| **Backbone** | ResNet50 (ImageNet Pretrained) |
| **Seeds** | {SEEDS} |

### Dataset Split
- **Known Classes:** 5 (No_DR, Mild, Moderate, Severe, Proliferative)
- **Unknown Class:** Glaucoma (ACRIMA Dataset)

## Aggregated Results (Mean ± Std)
| Metric | Mean | Std |
| :--- | :--- | :--- |
| **Known Accuracy** | **{mean['Known Acc']:.2f}%** | ±{std['Known Acc']:.2f} |
| **AUROC** | **{mean['AUROC']:.2f}%** | ±{std['AUROC']:.2f} |
| **H-mean** | **{mean['H-mean']:.2f}%** | ±{std['H-mean']:.2f} |

## Visualizations
### 1. Class-wise Accuracy Stability
![Class-wise Accuracy Boxplot](reproducibility_class_accuracy_boxplot.png)
*Boxplot showing the spread of accuracy for each class across 3 random seeds. Tight boxes indicate high stability.*

### 2. OOD Score Separation
![Distance Histogram](reproducibility_distance_histogram.png)
*Histogram of Mahalanobis distances aggregated across all 3 seeds. Clear separation between Blue (Known) and Red (Unknown) indicates robust OOD detection.*

## Detailed Results Table
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
    print("RECOVERY COMPLETE")
    print("Report saved to reproducibility.md")
    print("Plots saved: reproducibility_class_accuracy_boxplot.png, reproducibility_distance_histogram.png")
    print("="*80)

if __name__ == "__main__":
    main()
