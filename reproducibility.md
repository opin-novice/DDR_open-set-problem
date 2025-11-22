# Reproducibility Guide

## 1. Environment Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Run the Pipeline
Execute the master script:
```bash
python run_reproducibility_pipeline.py
```

This script will automatically:
1.  Train the model 3 times using random seeds `[42, 1, 2024]`.
2.  Save all outputs to `reproducibility_runs/seed_X/`.
3.  Generate a summary report `reproducibility_runs/final_report.txt` containing Mean ± Std for all metrics.

## 3. Artifacts
For each seed, the following files are saved in `reproducibility_runs/seed_X/`:
*   `model.pth`: The trained model checkpoint.
*   `train_list.txt`: List of training images and labels used.
*   `test_list.txt`: List of test images and labels used.

## 4. Configuration
Hyperparameters are documented in `config.yaml`.
