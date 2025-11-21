import os
import sys

def run():
    print("Starting DDR ARPL Training...")
    
    # Command configuration
    cmd = "python train_ddr_arpl.py"
    cmd += " --max_epoch 50"
    cmd += " --batch_size 16"
    cmd += " --lr 0.0001"
    cmd += " --train_class_num 3"
    cmd += " --test_class_num 5"
    cmd += " --eval_freq 1"
    cmd += " --outf checkpoints/ddr_resnet"
    
    print(f"Executing: {cmd}")
    os.system(cmd)
    
if __name__ == "__main__":
    run()
