import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*80}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*80}\n")
    # Use the current python executable
    ret = subprocess.call([sys.executable, script_name])
    if ret != 0:
        print(f"Error running {script_name}")
        exit(1)

if __name__ == "__main__":
    print("RESTORING WINNING STRATEGY: FOCAL LOSS + MAHALANOBIS")
    
    # Step 1: Train with Focal Loss (The winning model)
    # This script saves 'checkpoints/focal_closed_set.pth'
    run_script("train_focal_energy.py")
    
    # Step 2: Evaluate with Mahalanobis (The winning OOD method)
    # This script loads 'checkpoints/focal_closed_set.pth'
    run_script("mahalanobis_ood.py")
