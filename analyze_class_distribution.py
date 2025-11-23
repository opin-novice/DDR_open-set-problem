import numpy as np
import pandas as pd
from datasets import DDR
import torchvision.transforms as transforms

# Simple transform
transform = transforms.ToTensor()

# Load all splits
train_set = DDR('DDR dataset', split='train', transform=transform, train_class_num=5, test_class_num=5)
val_set = DDR('DDR dataset', split='val', transform=transform, train_class_num=5, test_class_num=5)
test_set = DDR('DDR dataset', split='test', transform=transform, train_class_num=5, test_class_num=5)

class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']

print("="*80)
print("CLASS DISTRIBUTION ANALYSIS")
print("="*80)

# Analyze each split
for split_name, dataset in [('TRAIN', train_set), ('VAL', val_set), ('TEST', test_set)]:
    print(f"\n{split_name} SET ({len(dataset)} samples):")
    print("-" * 60)
    
    labels = np.array(dataset.targets)
    
    for class_idx in range(5):
        count = (labels == class_idx).sum()
        percentage = count / len(labels) * 100
        print(f"  {class_names[class_idx]:15s}: {count:5d} samples ({percentage:5.2f}%)")

# Overall distribution
print("\n" + "="*80)
print("OVERALL DISTRIBUTION (All 25,044 images)")
print("="*80)

csv_path = 'DDR dataset/DR_grading.csv'
df = pd.read_csv(csv_path)
all_labels = df.iloc[:, 1].values

for class_idx in range(5):
    count = (all_labels == class_idx).sum()
    percentage = count / len(all_labels) * 100
    print(f"  {class_names[class_idx]:15s}: {count:5d} samples ({percentage:5.2f}%)")

# Check if stratification worked
print("\n" + "="*80)
print("STRATIFICATION CHECK")
print("="*80)

overall_dist = []
for class_idx in range(5):
    overall_dist.append((all_labels == class_idx).sum() / len(all_labels) * 100)

print("\nClass distribution should be similar across all splits:")
print(f"{'Class':<15s} {'Overall':<10s} {'Train':<10s} {'Val':<10s} {'Test':<10s}")
print("-" * 60)

for class_idx in range(5):
    overall_pct = (all_labels == class_idx).sum() / len(all_labels) * 100
    train_pct = (np.array(train_set.targets) == class_idx).sum() / len(train_set) * 100
    val_pct = (np.array(val_set.targets) == class_idx).sum() / len(val_set) * 100
    test_pct = (np.array(test_set.targets) == class_idx).sum() / len(test_set) * 100
    
    print(f"{class_names[class_idx]:<15s} {overall_pct:>9.2f}% {train_pct:>9.2f}% {val_pct:>9.2f}% {test_pct:>9.2f}%")

print("\n" + "="*80)
