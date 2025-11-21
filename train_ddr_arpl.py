import os
import argparse
import datetime
import time
import sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.models as models

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'OSR', 'ARPL'))

from datasets import DDR

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()
        
        # Forward pass
        logits = net(data)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Epoch: {}/{} Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(epoch+1, options['max_epoch'], batch_idx+1, len(trainloader), losses.val, losses.avg))

    return losses.avg

def evaluate(net, criterion, testloader, known_classes, **options):
    net.eval()
    correct = 0
    total = 0
    
    # For OSR metrics
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            logits = net(data)
            probs = F.softmax(logits, dim=1)
            
            # Predictions
            _, predicted = torch.max(logits, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    # 1. Known Class Accuracy
    known_mask = all_targets < len(known_classes)
    if np.sum(known_mask) > 0:
        # Get predictions for known samples
        known_probs = all_probs[known_mask]
        known_preds = np.argmax(known_probs, axis=1)
        known_targets_subset = all_targets[known_mask]
        known_acc = np.mean(known_preds == known_targets_subset) * 100
    else:
        known_acc = 0.0
        
    # 2. Unknown Detection (AUROC)
    # "Unknown" is defined as any class NOT in known_classes
    # We use Maximum Softmax Probability (MSP) as the confidence score.
    # Higher confidence => Likely Known
    # Lower confidence => Likely Unknown
    
    unknown_mask = all_targets >= len(known_classes)
    
    if np.sum(unknown_mask) > 0 and np.sum(known_mask) > 0:
        # Labels: 0 for Known, 1 for Unknown
        binary_labels = unknown_mask.astype(int)
        
        # Score: We want a score that is HIGH for Unknowns.
        # MSP is High for Knowns. So Score = 1 - MSP
        max_probs = np.max(all_probs, axis=1)
        unknown_scores = 1.0 - max_probs
        
        auroc = roc_auc_score(binary_labels, unknown_scores) * 100
    else:
        auroc = 0.0
        
    return known_acc, auroc

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: 
        use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Transforms
    # Standard ResNet50 transforms
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load Dataset
    print("Loading DDR dataset...")
    trainset = DDR(root=options['dataroot'], train=True, transform=transform_train,
                   train_class_num=options['train_class_num'], 
                   test_class_num=options['test_class_num'], 
                   includes_all_train_class=True)
    
    testset = DDR(root=options['dataroot'], train=False, transform=transform_test,
                  train_class_num=options['train_class_num'], 
                  test_class_num=options['test_class_num'], 
                  includes_all_train_class=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=options['batch_size'], 
                                              shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=options['batch_size'], 
                                             shuffle=False, num_workers=0)

    options['num_classes'] = options['train_class_num']
    print(f"Training on {options['num_classes']} known classes: {trainset.classes[:options['num_classes']]}")

    # Model: Standard ResNet50
    print("Creating model: Standard ResNet50 (Pretrained)")
    # Load pretrained ResNet50
    net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Replace FC layer
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, options['num_classes'])

    # Loss: Standard Cross Entropy
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    # Optimizer: AdamW is generally better for fine-tuning
    optimizer = torch.optim.AdamW(net.parameters(), lr=options['lr'], weight_decay=1e-4)
    
    if options['stepsize'] > 0:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=options['max_epoch'])

    options['use_gpu'] = use_gpu

    start_time = time.time()
    print(f"Starting training for {options['max_epoch']} epochs...")

    best_acc = 0.0

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        
        train_loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        print(f"Epoch {epoch+1} Loss: {train_loss:.4f}")

        if (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Testing")
            known_acc, auroc = evaluate(net, criterion, testloader, range(options['train_class_num']), **options)
            print(f"Epoch {epoch+1} Results:")
            print(f"  Known Class Accuracy:   {known_acc:.2f}%")
            print(f"  Unknown Detection AUROC: {auroc:.2f}%")
            
            if known_acc > best_acc:
                best_acc = known_acc
                # Save model
                save_path = os.path.join(options['outf'], 'best_resnet50_model.pth')
                torch.save(net.state_dict(), save_path)
                print(f"  -> Saved best model to {save_path}")

        if options['stepsize'] > 0: 
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    print(f"Best Known Class Accuracy: {best_acc:.2f}%")

    return best_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DDR ResNet50 Training")
    
    parser.add_argument('--dataroot', type=str, default='DDR dataset')
    parser.add_argument('--outf', type=str, default='checkpoints/ddr_resnet')
    parser.add_argument('--train_class_num', type=int, default=3)
    parser.add_argument('--test_class_num', type=int, default=5)
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--stepsize', type=int, default=30)
    
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    
    args = parser.parse_args()
    options = vars(args)
    
    if not os.path.exists(options['outf']):
        os.makedirs(options['outf'])
        
    main_worker(options)
