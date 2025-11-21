import os
import argparse
import datetime
import time
import csv
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from OSR.ARPL.models.models import classifier32, classifier32ABN
from datasets.ddr import DDR
import torchvision.transforms as transforms
from OSR.ARPL.utils import Logger, AverageMeter
from OSR.ARPL.core.test import test


# Custom FixRPLoss to address broadcasting issues
class FixRPLoss(nn.Module):
    def __init__(self, **options):
        super(FixRPLoss, self).__init__()
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.num_classes = options['num_classes']
        self.feat_dim = options['feat_dim']
        self.num_centers = options['num_centers']
        
        # Initialize centers properly
        self.centers = nn.Parameter(0.1 * torch.randn(self.num_classes * self.num_centers, self.feat_dim).cuda())
        self.radius = nn.Parameter(torch.Tensor([1.0]).cuda())

    def forward(self, features, labels=None):
        # Ensure features and centers are on the same device
        device = features.device
        
        # Compute distances between features and centers
        # features shape: [batch_size, feat_dim]
        # centers shape: [num_classes * num_centers, feat_dim]
        f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)  # [batch_size, 1]
        c_2 = torch.sum(torch.pow(self.centers, 2), dim=1, keepdim=True)  # [num_classes * num_centers, 1]
        # Broadcasting: [batch_size, 1] - [batch_size, num_classes*num_centers] + [1, num_classes*num_centers]
        dist = f_2 - 2 * torch.matmul(features, self.centers.t()) + c_2.t()
        dist = dist / float(features.shape[1])  # Normalize by feature dimension
        
        # Reshape to [batch_size, num_classes, num_centers]
        dist = torch.reshape(dist, [-1, self.num_classes, self.num_centers])
        dist = torch.mean(dist, dim=2)  # Average across centers: [batch_size, num_classes]
        
        # Compute logits
        logits = -dist  # Negative distance as logits (higher for closer points)
        
        if labels is None:
            return F.softmax(logits, dim=1), 0.0
            
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits / self.temp, labels)
        
        # Compute distance to assigned centers
        center_batch = self.centers[labels, :]  # [batch_size, feat_dim]
        _dis = (features - center_batch).pow(2).sum(1)  # [batch_size]
        loss_r = F.mse_loss(_dis, self.radius.expand_as(_dis))
        
        total_loss = ce_loss + self.weight_pl * loss_r
        
        return F.softmax(logits, dim=1), total_loss


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()

    torch.cuda.empty_cache()
    
    # Set model to training mode
    net.train()
    criterion.train()

    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            
            # Forward pass
            features, outputs = net(data, return_feature=True)  # Get both features and outputs
            logits, loss = criterion(features, labels)

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))

        if (batch_idx+1) % options['print_freq'] == 0:
            print("Epoch: {}/{} Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(epoch+1, options['max_epoch'], batch_idx+1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg

    return loss_all


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
        print("Currently using CPU (not recommended)")

    # Define transforms for the DDR dataset
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 to match classifier32 input
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 to match classifier32 input
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet normalization
    ])

    # Load DDR dataset
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
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=options['batch_size'], 
                                             shuffle=False, num_workers=4)

    options['num_classes'] = len(trainset.classes)
    print(f"Number of classes: {options['num_classes']} - {trainset.classes}")

    # Model
    print("Creating model: classifier32")
    net = classifier32(num_classes=options['num_classes'])
    feat_dim = 128  # From classifier32 architecture

    # Loss function
    options.update({
        'feat_dim': feat_dim,
        'use_gpu': use_gpu,
        'num_centers': options['num_centers']
    })

    criterion = FixRPLoss(**options)

    if use_gpu:
        print("Moving model and criterion to GPU...")
        net = net.cuda()  # Not using DataParallel to avoid potential broadcasting issues
        criterion = criterion.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    params_list = [
        {'params': net.parameters()},
        {'params': criterion.parameters()}
    ]

    # Use SGD optimizer as in the original ARPL
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90])

    start_time = time.time()
    print(f"Starting training for {options['max_epoch']} epochs...")

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        
        # Train the model
        train_loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        
        print(f"Epoch {epoch+1} - Training Loss: {train_loss:.6f}")

        # Test the model every eval_freq epochs or at the last epoch
        if (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Testing")
            # Note: test function would need to be adapted for DDR dataset
            # For now, we'll just use the train function with test data
            # In a real implementation, you'd want to implement proper testing
            net.eval()
            criterion.eval()
            
            correct = 0
            total = 0
            with torch.no_grad():
                for data, labels in testloader:
                    if use_gpu:
                        data, labels = data.cuda(), labels.cuda()
                    
                    features, outputs = net(data, return_feature=True)
                    logits, loss = criterion(features)
                    _, predicted = torch.max(logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} - Test Accuracy: {accuracy:.2f}%")

        if options['stepsize'] > 0: 
            scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished training. Total elapsed time (h:m:s): {}".format(elapsed))

    # Save the trained model
    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    file_name = f'ddr_fixrpl_{options["dataset"]}_{options["train_class_num"]}_{options["test_class_num"]}'
    
    # Save model and criterion
    torch.save(net.state_dict(), os.path.join(model_path, f'{file_name}_net.pth'))
    torch.save(criterion.state_dict(), os.path.join(model_path, f'{file_name}_criterion.pth'))
    
    print(f"Model saved to {model_path}")
    
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DDR OSR Training with FixRPLoss")
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='ddr', help="Dataset name for identification")
    parser.add_argument('--dataroot', type=str, default='../DDR dataset', help="Path to DDR dataset directory")
    parser.add_argument('--outf', type=str, default='./log', help="Output folder")
    parser.add_argument('--train_class_num', type=int, default=3, help="Number of known classes for training")
    parser.add_argument('--test_class_num', type=int, default=5, help="Total number of classes for testing")
    
    # Optimization
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size (16-32)")
    parser.add_argument('--lr', type=float, default=0.0005, help="Learning rate")
    parser.add_argument('--max-epoch', type=int, default=100, help="Number of epochs")
    parser.add_argument('--stepsize', type=int, default=30, help="Step size for learning rate decay")
    parser.add_argument('--temp', type=float, default=1.0, help="Temperature for softmax")
    parser.add_argument('--num-centers', type=int, default=1, help="Number of centers per class")
    
    # Model
    parser.add_argument('--weight-pl', type=float, default=0.1, help="Weight for pull loss")
    parser.add_argument('--model', type=str, default='classifier32', help="Model architecture")
    
    # Misc
    parser.add_argument('--eval-freq', type=int, default=10, help="Evaluation frequency")
    parser.add_argument('--print-freq', type=int, default=50, help="Print frequency")
    parser.add_argument('--gpu', type=str, default='0', help="GPU to use")
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--use-cpu', action='store_true', help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    options = vars(args)
    
    # Update options with default values if not provided
    options.setdefault('loss', 'FixRPLoss')
    
    print("Starting DDR OSR training with FixRPLoss...")
    print(f"Dataset root: {options['dataroot']}")
    print(f"Training classes: {options['train_class_num']}, Testing classes: {options['test_class_num']}")
    print(f"Batch size: {options['batch_size']}, LR: {options['lr']}, Epochs: {options['max_epoch']}")
    
    accuracy = main_worker(options)
    print(f"Training completed! Final accuracy: {accuracy:.2f}%")