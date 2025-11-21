# Applying Open Set Recognition to DDR Dataset (Diabetic Retinopathy Detection)

## Overview
The DDR dataset is a diabetic retinopathy detection dataset with 5 classes representing different stages of diabetic retinopathy:
- Class 0: No DR (No Diabetic Retinopathy)
- Class 1: Mild
- Class 2: Moderate  
- Class 3: Severe
- Class 4: Proliferative DR

## How to Apply Open Set Recognition to DDR Dataset

### 1. Dataset Structure
The DDR dataset should be organized as follows:
```
DDR dataset/
├── DR_grading.csv          # Contains image_id, diagnosis pairs
└── DR_grading/             # Folder containing actual image files
    ├── 20170413102628830.jpg
    ├── 20170413111955404.jpg
    └── ... (12,523 total images)
```

### 2. Using the DDR Dataset in OSR Experiments

The DDR dataset implements open set recognition functionality using the same interface as CIFAR and MNIST. You can use it as follows:

```python
from datasets import DDR
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Training set: Use only 3 known classes (e.g., No DR, Mild, Moderate)
trainset = DDR(root='./DDR dataset', train=True, 
               transform=transform,
               train_class_num=3,      # Known classes: [0, 1, 2]
               test_class_num=5,       # Total classes to test: [0, 1, 2, 3, 4] 
               includes_all_train_class=True)

# Testing set: Includes known classes [0, 1, 2] and unknown classes [3, 4]
testset = DDR(root='./DDR dataset', train=False, 
              transform=transform,
              train_class_num=3,      # Same as training
              test_class_num=5,       # Same as training
              includes_all_train_class=True)
```

### 3. Available OSR Algorithms

The project supports multiple open set recognition algorithms that can be applied to the DDR dataset:

#### A. OpenMax Algorithm
Location: `OSR/OpenMax/cifar100.py` (adapt for DDR)

Example usage:
```bash
python OSR/OpenMax/ddr.py --train_class_num 3 --test_class_num 5 --arch ResNet18
```

#### B. OLTR (Large-Scale Long-Tailed Recognition in an Open World)
Location: `OSR/OLTR/cifar100.py` (adapt for DDR)

Example usage:
```bash
python OSR/OLTR/ddr.py --train_class_num 3 --test_class_num 5 --arch ResNet18
```

#### C. DFP (Deep Feature Distribution Preserving)
Location: Various DFP implementations in `OSR/DFP*` directories

### 4. Implementation Details

The DDR dataset class implements open set functionality within the `_update_open_set()` method:

- **Training Phase**: Only samples from known classes (0 to `train_class_num`-1) are used
- **Testing Phase**: Samples from both known classes and unknown classes are included
- **Unknown Class Handling**: Unknown classes are mapped to a special "unknown" class index during evaluation
- **Openness Calculation**: During testing, openness metric is computed based on the ratio of unknown classes

### 5. Custom Training Script for DDR OSR

Create a custom training script at `OSR/DDR_experiment.py`:

```python
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from datasets import DDR
from Modelbuilder import Network  # or appropriate model builder

def train_ddr_osr():
    # Arguments
    train_class_num = 3  # Known classes: No DR, Mild, Moderate
    test_class_num = 5   # All classes: 0, 1, 2, 3, 4
    batch_size = 128
    epochs = 100
    learning_rate = 0.001

    # Transformations
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load datasets
    trainset = DDR(root='../DDR dataset', train=True, transform=transform_train,
                   train_class_num=train_class_num, test_class_num=test_class_num,
                   includes_all_train_class=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                             shuffle=True, num_workers=2)

    testset = DDR(root='../DDR dataset', train=False, transform=transform_test,
                  train_class_num=train_class_num, test_class_num=test_class_num,
                  includes_all_train_class=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    # Initialize network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Network(backbone='ResNet18', num_classes=train_class_num)
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        train(epoch, net, trainloader, optimizer, criterion, device)
        test(epoch, net, testloader, criterion, device)

def train(epoch, net, trainloader, optimizer, criterion, device):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        _, outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.3f}% ({correct}/{total})')

def test(epoch, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(f'Test Loss: {test_loss/(batch_idx+1):.3f} | Test Acc: {100.*correct/total:.3f}%')

if __name__ == '__main__':
    train_ddr_osr()
```

### 6. Evaluation Metrics

The DDR dataset will automatically calculate and output:
- **Openness**: Measures the proportion of unknown classes in the test set
- **Accuracy**: On known classes
- **F1-score**: Balanced metric for multiclass classification
- **ROC-AUC**: Area under ROC curve for binary open set classification

### 7. Key Considerations

1. **Class Imbalance**: The DDR dataset is naturally imbalanced, with different numbers of samples per class
2. **Image Quality**: Images vary in quality and may need preprocessing
3. **Feature Extraction**: Deep features from ResNet/EfficientNet backbones work well for this medical imaging task
4. **Threshold Tuning**: For methods like OpenMax, threshold tuning is critical for separating known and unknown classes

### 8. Expected Results

With proper OSR techniques, you should expect:
- High accuracy on known classes during closed-set evaluation
- Good separation between known and unknown classes during open-set evaluation
- Improved performance of OpenMax, OLTR, or DFP compared to baseline softmax classifier