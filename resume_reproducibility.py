import os
import numpy as np
import pandas as pd
import torch
import sys

# Import functions from our scripts
from train_focal_energy import train_with_focal_loss
from mahalanobis_ood import main as eval_main

def run_pipeline():
    SEEDS = [42, 1, 2024]
    metrics = []
    
    print("="*80)
    print("RESUMING REPRODUCIBILITY PIPELINE")
    print(f"Seeds: {SEEDS}")
    print("="*80)
    
    for seed in SEEDS:
        print(f"\n\n>>> PROCESSING SEED {seed} <<<")
        output_dir = f"reproducibility_runs/seed_{seed}"
        model_path = os.path.join(output_dir, 'model.pth')
        
        # 1. Check if training is already done
        if os.path.exists(model_path):
            print(f"Found existing model at {model_path}. Skipping training.")
        else:
            print(f"No model found at {model_path}. Starting training...")
            # Train
            train_with_focal_loss(
                dataroot='DDR dataset',
                num_epochs=40,
                batch_size=32,
                lr=0.001,
                seed=seed,
                output_dir=output_dir
            )
        
        # 2. Evaluate
        print(f"\n>>> EVALUATING SEED {seed} <<<")
        # Ensure model exists now
        if not os.path.exists(model_path):
             print(f"ERROR: Model not found even after training step for seed {seed}. Skipping evaluation.")
             continue

        stats = eval_main(model_path=model_path)
        
        # Flatten stats for dataframe
        row = {
            'Seed': seed,
            'Known Acc': stats['known_acc'],
            'AUROC': stats['auroc'],
            'Combined': stats['combined'],
            'Class 0 Acc': stats['class_acc'].get(0, 0),
            'Class 1 Acc': stats['class_acc'].get(1, 0),
            'Class 2 Acc': stats['class_acc'].get(2, 0)
        }
        metrics.append(row)
        
    # 3. Aggregate Results
    if not metrics:
        print("No metrics collected.")
        return

    df = pd.DataFrame(metrics)
    
    print("\n\n" + "="*80)
    print("REPRODUCIBILITY REPORT")
    print("="*80)
    print(df.to_string(index=False, float_format="%.2f"))
    
    print("\n" + "-"*80)
    print("SUMMARY STATISTICS (Mean ± Std)")
    print("-" * 80)
    
    summary = df.describe().loc[['mean', 'std']]
    print(summary.to_string(float_format="%.2f"))
    
    # Save report
    os.makedirs('reproducibility_runs', exist_ok=True)
    df.to_csv('reproducibility_runs/metrics.csv', index=False)
    
    with open('reproducibility_runs/final_report.txt', 'w') as f:
        f.write("REPRODUCIBILITY REPORT\n")
        f.write("======================\n\n")
        f.write(df.to_string(index=False, float_format="%.2f"))
        f.write("\n\nSUMMARY STATISTICS\n")
        f.write("------------------\n")
        f.write(summary.to_string(float_format="%.2f"))
        
    print(f"\nReport saved to reproducibility_runs/final_report.txt")

if __name__ == "__main__":
    run_pipeline()
