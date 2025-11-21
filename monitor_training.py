"""
Training Monitor - Periodically checks training progress and reports updates
"""
import os
import time
import sys

def monitor_training(history_file, checkpoint_dir, update_interval=300):
    """
    Monitor training progress every 'update_interval' seconds (default 5 minutes)
    """
    
    last_epoch_reported = 0
    training_start = time.time()
    
    print("="*80)
    print("TRAINING MONITOR STARTED")
    print(f"History file: {history_file}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Update interval: {update_interval}s ({update_interval/60:.1f} min)")
    print("="*80)
    
    while True:
        try:
            # Wait for interval
            time.sleep(update_interval)
            
            # Check if history file exists
            if not os.path.exists(history_file):
                elapsed = time.time() - training_start
                print(f"\n[{elapsed/60:.1f} min] Waiting for training to start...")
                continue
            
            # Read history file
            with open(history_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) <= 1:  # Only header
                continue
            
            # Get latest epoch
            latest_line = lines[-1].strip()
            parts = latest_line.split(',')
            
            if len(parts) < 4:
                continue
            
            epoch = int(parts[0])
            train_loss = float(parts[1])
            known_acc = float(parts[2])
            auroc = float(parts[3])
            
            # Only report if new epoch
            if epoch > last_epoch_reported:
                elapsed = time.time() - training_start
                
                print("\n" + "="*80)
                print(f"PROGRESS UPDATE - Epoch {epoch}")
                print("="*80)
                print(f"Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
                print(f"Train Loss:           {train_loss:.4f}")
                print(f"Known Class Accuracy: {known_acc:.2f}%")
                print(f"Unknown Detection AUROC: {auroc:.2f}%")
                
                if known_acc > 0 and auroc > 0:
                    combined = 2 * (known_acc * auroc) / (known_acc + auroc)
                    print(f"Combined Score:       {combined:.2f}%")
                
                # Estimate time remaining (rough)
                if epoch > 0:
                    time_per_epoch = elapsed / epoch
                    remaining_epochs = 50 - epoch  # Assuming 50 total epochs
                    eta_seconds = time_per_epoch * remaining_epochs
                    print(f"\nEstimated time remaining: {eta_seconds/3600:.2f} hours")
                
                print("="*80)
                
                last_epoch_reported = epoch
                
                # Check if training seems stuck
                if train_loss > 1.5 and epoch > 10:
                    print("\n⚠️  WARNING: Training loss seems high after 10 epochs")
                
                if known_acc < 50 and epoch > 10:
                    print("\n⚠️  WARNING: Known accuracy still low after 10 epochs")
                
        except KeyboardInterrupt:
            print("\n\nMonitor stopped by user.")
            break
        except Exception as e:
            print(f"\nMonitor error: {e}")
            continue

if __name__ == '__main__':
    history_file = 'checkpoints/ddr_arpl_improved/training_history.txt'
    checkpoint_dir = 'checkpoints/ddr_arpl_improved'
    
    # Update every 5 minutes
    monitor_training(history_file, checkpoint_dir, update_interval=300)
