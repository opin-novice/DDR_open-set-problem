import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_with_focal_loss(
    dataroot='DDR dataset',
    num_epochs=50,
    batch_size=32,
    lr=0.001,
    output_dir='checkpoints',
    patience=10  # Increased from 5 to 10
):
    print("="*80)
    print("FINAL EXPERT-TUNED TRAINING: FOCAL LOSS + AGGRESSIVE OVERSAMPLING")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Split: 80/10/10 (Train/Val/Test)")
    print(f"  - Focal Loss: α=0.25, γ=3.0 (AGGRESSIVE)")
    print(f"  - Mild Oversampling: 20x (CRITICAL BOOST)")
    print(f"  - Severe Oversampling: 30x (MAXIMUM BOOST)")
    print(f"  - Epochs: {num_epochs} | Early Stopping: Patience 10 (after warmup)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Data Setup
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Enhanced augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load DDR dataset with 80/10/10 split
    print("\nLoading DDR Dataset (5 Classes) with 80/10/10 split...")
    trainset = DDR(root=dataroot, split='train', transform=transform_train, 
                   train_class_num=5, test_class_num=5, includes_all_train_class=True)
    valset = DDR(root=dataroot, split='val', transform=transform_test, 
                 train_class_num=5, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root=dataroot, split='test', transform=transform_test, 
                  train_class_num=5, test_class_num=5, includes_all_train_class=True)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(trainset)} (80%)")
    print(f"Validation samples: {len(valset)} (10%)")
    print(f"Testing samples: {len(testset)} (10%)")
    
    # Class distribution analysis
    train_labels = np.array(trainset.targets)
    class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print("\nClass Distribution in Training Set:")
    for i, name in enumerate(class_names):
        count = (train_labels == i).sum()
        pct = count / len(train_labels) * 100
        print(f"  {name:15s}: {count:5d} samples ({pct:5.2f}%)")
    
    # *** EXPERT-TUNED CLASS-SPECIFIC OVERSAMPLING ***
    print("\n🔧 Implementing AGGRESSIVE class-specific oversampling...")
    print("   Strategy: Target minority classes with clinical precision")
    
    # Calculate base class frequencies
    class_counts = np.bincount(train_labels)
    
    # Expert-defined oversampling factors (based on clinical importance + sample size)
    # No_DR: 1x (majority class, no oversampling needed)
    # Mild: 20x (most problematic, needs aggressive boost)
    # Moderate: 1.4x (second majority, slight boost)
    # Severe: 30x (smallest class, maximum boost)
    # Proliferative: 7x (small class, moderate boost)
    
    oversample_factors = np.array([1.0, 20.0, 1.4, 30.0, 7.0])
    
    # Calculate sample weights
    sample_weights = np.zeros(len(train_labels))
    for class_idx in range(5):
        class_mask = train_labels == class_idx
        sample_weights[class_mask] = oversample_factors[class_idx]
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    
    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Recreate trainloader with sampler
    trainloader = DataLoader(trainset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    
    print("✅ Class-specific oversampling enabled:")
    for i, name in enumerate(class_names):
        effective_samples = int(class_counts[i] * oversample_factors[i])
        print(f"  {name:15s}: {oversample_factors[i]:5.1f}x → ~{effective_samples:5d} samples/epoch")
    print("\n  🎯 Mild class will appear 20x more often (critical for recovery)")
    print("  🎯 Severe class will appear 30x more often (maximum boost)")
    
    
    
    # 2. Model Setup
    print("\nInitializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 5)
    model = model.to(device)
    
    # 3. Focal Loss and Optimizer (EXPERT-TUNED)
    print("\n🔬 Initializing AGGRESSIVE Focal Loss (γ=3.0)...")
    criterion = FocalLoss(alpha=0.25, gamma=3.0)  # Increased from 2.0 to 3.0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 4. Training Loop with Early Stopping
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        class_correct = np.zeros(5)
        class_total = np.zeros(5)
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(5):
                mask = labels == i
                if mask.sum() > 0:
                    class_total[i] += mask.sum().item()
                    class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
        
        train_acc = 100.0 * train_correct / train_total
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_class_correct = np.zeros(5)
        val_class_total = np.zeros(5)
        
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                for i in range(5):
                    mask = labels == i
                    if mask.sum() > 0:
                        val_class_total[i] += mask.sum().item()
                        val_class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        
        # Calculate per-class accuracies
        train_class_acc = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                          for i in range(5)]
        val_class_acc = [100.0 * val_class_correct[i] / val_class_total[i] if val_class_total[i] > 0 else 0 
                         for i in range(5)]
        
        train_history.append(train_acc)
        val_history.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"  Loss: {running_loss/len(trainloader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        print(f"  Per-Class Val Accuracy:")
        for i, name in enumerate(class_names):
            status = "⚠️" if val_class_acc[i] < 30 else "✓"
            print(f"    {status} {name:15s}: {val_class_acc[i]:5.2f}%")
        
        # Alert for Severe class
        if val_class_acc[3] < 30:  # Severe is index 3
            print(f"  🚨 ALERT: Severe class accuracy is critically low ({val_class_acc[3]:.2f}%)")
        
        # Early stopping check (with warmup period)
        if epoch >= 10:  # Don't trigger early stopping before epoch 10
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                save_path = os.path.join(output_dir, 'resnet50_full_5class.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  >>> New Best Model Saved! (Val Acc: {best_val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/10")
                
            if patience_counter >= 10:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                print(f"  Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
                break
        else:
            # During warmup, always save if better
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                save_path = os.path.join(output_dir, 'resnet50_full_5class.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  >>> New Best Model Saved! (Val Acc: {best_val_acc:.2f}%)")
            print(f"  [Warmup: {epoch+1}/10 - Early stopping disabled]")
    
    # 5. Final Test Set Evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(output_dir, 'resnet50_full_5class.pth')))
    model.eval()
    
    test_correct = 0
    test_total = 0
    test_class_correct = np.zeros(5)
    test_class_total = np.zeros(5)
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            for i in range(5):
                mask = labels == i
                if mask.sum() > 0:
                    test_class_total[i] += mask.sum().item()
                    test_class_correct[i] += (predicted[mask] == labels[mask]).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    test_class_acc = [100.0 * test_class_correct[i] / test_class_total[i] if test_class_total[i] > 0 else 0 
                     for i in range(5)]
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title(f'Confusion Matrix (Test Acc: {test_acc:.2f}%)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('focal_loss_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Final Report
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nPer-Class Test Accuracy:")
    for i, name in enumerate(class_names):
        status = "✓" if test_class_acc[i] >= 50 else "⚠️" if test_class_acc[i] >= 30 else "🚨"
        print(f"  {status} {name:15s}: {test_class_acc[i]:5.2f}% ({int(test_class_total[i])} samples)")
    
    # Save detailed report
    with open('focal_loss_training_report.txt', 'w') as f:
        f.write("Focal Loss Training Report (80/10/10 Split)\n")
        f.write("="*60 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Test Accuracy: {test_acc:.2f}%\n")
        f.write(f"Training stopped at epoch: {best_epoch}\n\n")
        f.write("Per-Class Test Accuracy:\n")
        for i, name in enumerate(class_names):
            f.write(f"  {name:15s}: {test_class_acc[i]:5.2f}%\n")
    
    print(f"\nModel saved to: {os.path.join(output_dir, 'resnet50_full_5class.pth')}")
    print(f"Confusion matrix saved to: focal_loss_confusion_matrix.png")
    print(f"Report saved to: focal_loss_training_report.txt")
    print("="*80)

if __name__ == "__main__":
    train_with_focal_loss()
