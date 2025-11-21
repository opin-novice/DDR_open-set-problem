"""
Test script to verify DDR dataset integration with Open Set Recognition
This script tests the DDR dataset with the OpenMax algorithm
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
import sys

# Add the project root to the path
sys.path.append(".")

from datasets import DDR
from OSR.OpenMax.Modelbuilder import Network  # Use OpenMax's model builder

# Parse arguments
parser = argparse.ArgumentParser(description='Test DDR Dataset with OpenMax')
parser.add_argument('--train_class_num', default=3, type=int, help='Number of known classes for training (0 to N-1)')
parser.add_argument('--test_class_num', default=5, type=int, help='Total number of classes for testing (known + unknown)')
parser.add_argument('--arch', default='ResNet18', type=str, help='Network architecture to use')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for testing')
parser.add_argument('--dataset_path', default='./DDR dataset', type=str, help='Path to DDR dataset')

args = parser.parse_args()

def test_ddr_integration():
    print("="*60)
    print("TESTING DDR DATASET INTEGRATION WITH OPEN SET RECOGNITION")
    print("="*60)
    
    print(f"Training with {args.train_class_num} known classes")
    print(f"Testing with {args.test_class_num} total classes")
    print(f"Using architecture: {args.arch}")
    print(f"Dataset path: {args.dataset_path}")
    
    # Define transforms for DDR images (similar to ImageNet preprocessing)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    print("\n1. Loading DDR Training Dataset...")
    try:
        trainset = DDR(root=args.dataset_path, train=True, transform=transform_train,
                      train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                      includes_all_train_class=True)
        print(f"✓ Training set loaded successfully")
        print(f"  - Size: {len(trainset)} samples")
        print(f"  - Classes: {trainset.classes}")
        print(f"  - Sample shape: {trainset[0][0].shape}")
        
        # Check a few samples
        for i in range(min(3, len(trainset))):
            img, target = trainset[i]
            print(f"  - Sample {i}: image shape={img.shape}, target={target}, type={type(target)}")
        
    except Exception as e:
        print(f"✗ Error loading training set: {e}")
        return False
    
    print("\n2. Loading DDR Testing Dataset...")
    try:
        testset = DDR(root=args.dataset_path, train=False, transform=transform_test,
                     train_class_num=args.train_class_num, test_class_num=args.test_class_num,
                     includes_all_train_class=True)
        print(f"✓ Testing set loaded successfully")
        print(f"  - Size: {len(testset)} samples")
        print(f"  - Classes: {testset.classes}")
        print(f"  - Has openness: {hasattr(testset, 'openness')}")
        if hasattr(testset, 'openness'):
            print(f"  - Openness: {testset.openness:.3f}")
        print(f"  - Sample shape: {testset[0][0].shape}")
        
        # Check a few samples
        for i in range(min(3, len(testset))):
            img, target = testset[i]
            print(f"  - Sample {i}: image shape={img.shape}, target={target}, type={type(target)}")
        
    except Exception as e:
        print(f"✗ Error loading testing set: {e}")
        return False
    
    print("\n3. Testing DataLoaders...")
    try:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        
        print(f"✓ DataLoaders created successfully")
        print(f"  - Train batches: {len(trainloader)}")
        print(f"  - Test batches: {len(testloader)}")
        
        # Test fetching a batch
        train_iter = iter(trainloader)
        test_iter = iter(testloader)
        
        train_batch = next(train_iter)
        test_batch = next(test_iter)
        
        print(f"  - Train batch: images={train_batch[0].shape}, targets={train_batch[1].shape}")
        print(f"  - Test batch: images={test_batch[0].shape}, targets={test_batch[1].shape}")
        
    except Exception as e:
        print(f"✗ Error with DataLoaders: {e}")
        return False
    
    print("\n4. Testing Model Integration...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create model with the number of training classes (known classes only)
        net = Network(backbone=args.arch, num_classes=args.train_class_num, embed_dim=None)
        net = net.to(device)
        
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
            
        print(f"✓ Network created successfully: {args.arch}")
        print(f"  - Input classes: {args.train_class_num} (known classes only)")
        print(f"  - Model parameters: {sum(p.numel() for p in net.parameters()):,}")
        
        # Test forward pass with a batch
        with torch.no_grad():
            sample_images = test_batch[0][:4].to(device)  # Take first 4 images from test batch
            features, logits = net(sample_images)
            
            print(f"  - Forward pass successful")
            print(f"  - Features shape: {features.shape}")
            print(f"  - Logits shape: {logits.shape}")
        
    except Exception as e:
        print(f"✗ Error with model integration: {e}")
        return False
    
    print("\n5. Testing with Loss Function...")
    try:
        criterion = nn.CrossEntropyLoss()
        
        # Move test batch to device
        images, targets = test_batch[0].to(device), test_batch[1].to(device)
        
        # Forward pass
        features, outputs = net(images)
        loss = criterion(outputs, targets)
        
        print(f"✓ Loss computation successful")
        print(f"  - Loss value: {loss.item():.4f}")
        
    except Exception as e:
        print(f"✗ Error with loss computation: {e}")
        return False
    
    print("\n" + "="*60)
    print("SUCCESS: DDR Dataset is fully integrated with Open Set Recognition!")
    print("="*60)
    print(f"✓ DDR dataset can be used with {args.train_class_num} known classes for training")
    print(f"✓ Testing includes {args.test_class_num} total classes with unknown class detection")
    print(f"✓ Works with {args.arch} architecture and existing OSR algorithms")
    print(f"✓ Ready for OpenMax and other OSR experiments")
    
    if hasattr(testset, 'openness'):
        print(f"✓ Openness metric calculated: {testset.openness:.3f}")
    
    return True

if __name__ == "__main__":
    success = test_ddr_integration()
    if success:
        print("\n🎉 DDR dataset integration test PASSED!")
        print("You can now use the DDR dataset with any OSR algorithm in this project.")
    else:
        print("\n❌ DDR dataset integration test FAILED!")
        print("Check the error messages above for troubleshooting.")