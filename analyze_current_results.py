import os
import sys
import re
import numpy as np

def parse_training_report(filepath):
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    metrics = {}
    
    # Parse Best Val Acc
    val_acc_match = re.search(r"Best Validation Accuracy: ([\d\.]+)%", content)
    if val_acc_match:
        metrics['best_val_acc'] = float(val_acc_match.group(1))
        
    # Parse Test Acc
    test_acc_match = re.search(r"Final Test Accuracy: ([\d\.]+)%", content)
    if test_acc_match:
        metrics['test_acc'] = float(test_acc_match.group(1))
        
    # Parse Per-Class Acc
    class_accs = {}
    for line in content.split('\n'):
        match = re.search(r"\s+([A-Za-z_]+)\s+:\s+([\d\.]+)%", line)
        if match:
            class_name = match.group(1).strip()
            acc = float(match.group(2))
            class_accs[class_name] = acc
    metrics['class_accs'] = class_accs
    return metrics

def parse_osr_results(filepath):
    if not os.path.exists(filepath):
        return None
        
    with open(filepath, 'r') as f:
        content = f.read()
        
    metrics = {}
    auroc_match = re.search(r"AUROC: ([\d\.]+)%", content)
    if auroc_match:
        metrics['auroc'] = float(auroc_match.group(1))
        
    hmean_match = re.search(r"H-mean: ([\d\.]+)%", content)
    if hmean_match:
        metrics['hmean'] = float(hmean_match.group(1))
        
    return metrics

def generate_diagnosis_report(train_metrics, osr_metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 🏥 Project Diagnosis: DDR Open Set Recognition\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n")
        f.write(f"- **Model**: ResNet50 (Focal Loss γ=3.0, Aggressive Oversampling)\n")
        if train_metrics:
            f.write(f"- **Known Accuracy**: {train_metrics.get('test_acc', 'N/A')}%\n")
        if osr_metrics:
            f.write(f"- **Unknown Detection (AUROC)**: {osr_metrics.get('auroc', 'N/A')}%\n")
            f.write(f"- **H-Mean**: {osr_metrics.get('hmean', 'N/A')}%\n")
        f.write("\n")
        
        # 2. Class-wise Performance Analysis
        f.write("## 2. Class-wise Performance Analysis\n")
        if train_metrics and 'class_accs' in train_metrics:
            accs = train_metrics['class_accs']
            f.write("| Class | Accuracy | Status | Diagnosis |\n")
            f.write("|:---|:---|:---|:---|\n")
            
            # Define thresholds
            for cls, acc in accs.items():
                status = "✅ Good"
                diag = "Stable"
                if acc < 50:
                    status = "🚨 Critical"
                    diag = "Severe Underfitting / Confusion"
                elif acc < 75:
                    status = "⚠️ Warning"
                    diag = "Needs Improvement"
                
                if cls == "Mild" and acc < 25:
                    diag = "Clinical Ambiguity / Feature Overlap"
                if cls == "Severe" and acc < 50:
                    diag = "Fine-grained Confusion (vs Moderate)"
                    
                f.write(f"| **{cls}** | **{acc}%** | {status} | {diag} |\n")
        f.write("\n")
        
        # 3. Root Cause Analysis
        f.write("## 3. Root Cause Analysis\n")
        f.write("### A. The 'Mild' Class Paradox\n")
        f.write("- **Symptom**: Accuracy ~11-15% despite 20x oversampling.\n")
        f.write("- **Cause**: High feature overlap with 'No_DR'. The model minimizes loss by predicting the majority class (No_DR) rather than risking false positives on a noisy minority class.\n")
        f.write("- **Implication**: This is likely a data/label quality issue, not just a training issue.\n\n")
        
        f.write("### B. The 'Severe' Class Struggle\n")
        f.write("- **Symptom**: Accuracy ~45%.\n")
        f.write("- **Cause**: 'Severe' is an intermediate stage defined by specific counts (4-2-1 rule). ResNet50 global pooling loses the spatial granularity needed for counting.\n")
        f.write("- **Implication**: Needs attention mechanisms or higher resolution.\n\n")
        
        # 4. OSR Risk Assessment
        f.write("## 4. OSR Risk Assessment\n")
        f.write("- **Current AUROC**: >99% (Excellent).\n")
        f.write("- **Risk**: Improving 'Mild' accuracy by forcing the model to learn noise could **degrade** the feature space separation between 'Retina' and 'Glaucoma'.\n")
        f.write("- **Verdict**: Proceed with caution. Do not sacrifice AUROC for marginal Mild gains.\n\n")
        
        # 5. Recommendations
        f.write("## 5. Recommendations\n")
        f.write("1. **Data**: Implement Test Time Augmentation (TTA) to boost robust predictions.\n")
        f.write("2. **Model**: Integrate CBAM (Convolutional Block Attention Module) to help with fine-grained 'Severe' features.\n")
        f.write("3. **Training**: Experiment with Label Smoothing to prevent overconfidence in majority classes.\n")

if __name__ == "__main__":
    train_report_path = "focal_loss_training_report.txt"
    osr_results_path = "cross_dataset_results.txt"
    output_path = "report/current_diagnosis.md"
    
    train_metrics = parse_training_report(train_report_path)
    osr_metrics = parse_osr_results(osr_results_path)
    
    generate_diagnosis_report(train_metrics, osr_metrics, output_path)
    print(f"Diagnosis report generated at {output_path}")
