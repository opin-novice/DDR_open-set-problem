import os
import pandas as pd
import numpy as np
from datasets.ddr import DDR
import torch

def check_distribution():
    print("Checking DDR Dataset Distribution...")
    
    # 1. Check CSV
    csv_path = 'DDR dataset/DR_grading.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\nCSV loaded from {csv_path}")
        print(f"Total samples: {len(df)}")
        print("Class distribution in CSV:")
        counts = df.iloc[:, 1].value_counts().sort_index()
        for label, count in counts.items():
            print(f"  Class {label}: {count}")
    else:
        print(f"\nCSV not found at {csv_path}")
        return

    # 2. Check Dataset Split
    print("\nInitializing DDR Test Set...")
    try:
        ds = DDR(root='DDR dataset', split='test', 
                 train_class_num=5, test_class_num=5, includes_all_train_class=True)
        
        print(f"\nTest Set Size: {len(ds)}")
        targets = np.array(ds.targets)
        unique, counts = np.unique(targets, return_counts=True)
        print("Class distribution in Test Set:")
        for label, count in zip(unique, counts):
            class_name = ds.classes[label] if label < len(ds.classes) else f"Unknown({label})"
            print(f"  Class {label} ({class_name}): {count}")
            
    except Exception as e:
        print(f"Error initializing dataset: {e}")

if __name__ == "__main__":
    check_distribution()
