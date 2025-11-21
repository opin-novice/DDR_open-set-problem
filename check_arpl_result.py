import torch
import os

try:
    path = 'checkpoints/ddr_arpl_v3/best_combined_model.pth'
    if os.path.exists(path):
        checkpoint = torch.load(path)
        print(f"Epoch: {checkpoint.get('epoch')}")
        print(f"Known Acc: {checkpoint.get('known_acc'):.2f}%")
        print(f"AUROC: {checkpoint.get('auroc'):.2f}%")
    else:
        print("Checkpoint not found.")
except Exception as e:
    print(f"Error: {e}")
