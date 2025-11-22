import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR

def train_full_ddr(
    dataroot='DDR dataset',
    num_epochs=30,
    batch_size=32,
    lr=0.001,
    output_dir='checkpoints'
):
    print("="*80)
    print("TRAINING FULL 5-CLASS DDR MODEL")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Data Setup
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Load FULL DDR dataset (train_class_num=5)
    print("\nLoading DDR Dataset (5 Classes)...")
    trainset = DDR(root=dataroot, train=True, transform=transform_train, 
                   train_class_num=5, test_class_num=5, includes_all_train_class=True)
    testset = DDR(root=dataroot, train=False, transform=transform_test, 
                  train_class_num=5, test_class_num=5, includes_all_train_class=True)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training samples: {len(trainset)}")
    print(f"Testing samples: {len(testset)}")
    
    # 2. Model Setup
    print("\nInitializing ResNet50...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    # Change output to 5 classes
    model.fc = nn.Linear(num_ftrs, 5)
    model = model.to(device)
    
    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_acc = 100.0 * correct / total
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {running_loss/len(trainloader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(output_dir, 'resnet50_full_5class.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Model Saved! ({best_acc:.2f}%)")
            
    print("\n" + "="*80)
    print(f"Training Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {os.path.join(output_dir, 'resnet50_full_5class.pth')}")
    print("="*80)

if __name__ == "__main__":
    train_full_ddr()
