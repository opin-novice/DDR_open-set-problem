from __future__ import print_function
from torchvision.datasets.vision import VisionDataset
import warnings
from PIL import Image
import os
import os.path
import numpy as np
import torch
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive


class DDR(VisionDataset):
    """DDR Diabetic Retinopathy Detection Dataset

    Args:
        root (string): Root directory of dataset where directory
            ``DDR/processed/training.pt`` and  ``DDR/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        train_class_num (int): Number of known classes to use for training
        test_class_num (int): Number of classes to include in testing (including known and unknown)
        includes_all_train_class (bool): If contains all unknown classes in testing.
    """

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, split=None, transform=None, target_transform=None,
                 download=False, train_class_num=3, test_class_num=5, includes_all_train_class=True):
        super(DDR, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        
        # Handle backward compatibility: if split is None, use train parameter
        if split is None:
            split = 'train' if train else 'test'
        
        self.split = split  # 'train', 'val', or 'test'
        self.train = (split == 'train')  # For backward compatibility
        self.train_class_num = train_class_num
        self.test_class_num = test_class_num
        self.includes_all_train_class = includes_all_train_class

        # Load the CSV file containing the image paths and labels
        csv_path = os.path.join(self.root, 'DR_grading.csv')
        if not os.path.exists(csv_path):
            raise RuntimeError('DDR dataset CSV file not found. You can use download=True to download it '
                               'or put it in the root directory.')

        dataframe = pd.read_csv(csv_path)
        all_image_paths = dataframe.iloc[:, 0].values  # First column contains image IDs
        all_targets = dataframe.iloc[:, 1].values     # Second column contains labels

        # Convert targets to integers
        all_targets = all_targets.astype(np.int64)

        # Define classes (0 to 4 for DDR - representing different stages of diabetic retinopathy)
        self.classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

        # Implement 80/10/10 train/val/test split with stratification
        from sklearn.model_selection import train_test_split
        
        # Split into train+val (90%) and test (10%)
        train_val_idx, test_idx = train_test_split(
            np.arange(len(all_targets)),
            test_size=0.10,
            stratify=all_targets,
            random_state=42  # Fixed seed for reproducibility
        )
        
        # Split train+val into train (80% of total = 88.89% of train+val) and val (10% of total)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.10/0.90,  # 10% of total / 90% of total = ~11.11% of train+val
            stratify=all_targets[train_val_idx],
            random_state=42
        )
        
        # Select data based on split
        if self.split == 'train':
            selected_idx = train_idx
        elif self.split == 'val':
            selected_idx = val_idx
        elif self.split == 'test':
            selected_idx = test_idx
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'")
        
        self.image_paths = all_image_paths[selected_idx]
        self.targets = all_targets[selected_idx]

        # Initialize dataset attributes
        self.train_class_num_actual = train_class_num
        self.test_class_num_actual = test_class_num

        # Apply open set transformation
        self._update_open_set(train_class_num, test_class_num, includes_all_train_class)

    def _update_open_set(self, train_class_num=3, test_class_num=5, includes_all_train_class=True):
        """
        Update the dataset to simulate open set recognition scenario
        :param train_class_num: Number of known classes for training
        :param test_class_num: Number of classes to include in testing
        :param includes_all_train_class: Whether to include all training classes in test
        """
        print(f"Original classes: {self.classes}")
        print(f"Training with {train_class_num} known classes, Testing with {test_class_num} total classes")

        # Define class list (0 to 4 for DDR dataset)
        class_list = list(range(len(self.classes)))

        # Select training classes (typically 0 to train_class_num-1)
        train_classes = list(range(train_class_num))

        # For testing, select test_class_num classes
        if includes_all_train_class:
            # Include all training classes plus additional unknown classes
            test_classes = train_classes + list(range(train_class_num, test_class_num))
        else:
            # Randomly select classes
            test_classes = list(range(test_class_num))

        # Update classes to include 'unknown' label
        selected_elements = [self.classes[index] for index in train_classes]
        selected_elements.append('unknown')
        self.classes = selected_elements

        print(f"Updated classes: {self.classes}")

    def _update_open_set(self, train_class_num=3, test_class_num=5, includes_all_train_class=True):
        """
        Update the dataset to simulate open set recognition scenario
        :param train_class_num: Number of known classes for training
        :param test_class_num: Number of classes to include in testing
        :param includes_all_train_class: Whether to include all training classes in test
        """
        print(f"Original classes: {self.classes}")
        print(f"Split: {self.split}, Training with {train_class_num} known classes, Testing with {test_class_num} total classes")

        # Define class list (0 to 4 for DDR dataset)
        class_list = list(range(len(self.classes)))

        # Select training classes (typically 0 to train_class_num-1)
        train_classes = list(range(train_class_num))

        # For testing, select test_class_num classes
        if includes_all_train_class:
            # Include all training classes plus additional unknown classes
            test_classes = train_classes + list(range(train_class_num, test_class_num))
        else:
            # Randomly select classes
            test_classes = list(range(test_class_num))

        # Update classes to include 'unknown' label
        selected_elements = [self.classes[index] for index in train_classes]
        selected_elements.append('unknown')
        self.classes = selected_elements

        print(f"Updated classes: {self.classes}")

        # Create boolean mask for selecting samples based on split
        # For train and val: only use samples from training classes
        # For test: use samples from test classes
        if self.split in ['train', 'val']:
            # Training/Validation phase: only use samples from training classes
            indexes = [i for i, x in enumerate(self.targets) if int(x) in train_classes]
        else:
            # Testing phase: use samples from test classes
            indexes = [i for i, x in enumerate(self.targets) if int(x) in test_classes]

        # Update the data and targets based on selected indexes
        self.image_paths = [self.image_paths[i] for i in indexes]
        self.targets = [int(self.targets[i]) for i in indexes]

        # Store class numbers as instance variables
        self.train_class_num_actual = train_class_num
        self.test_class_num_actual = test_class_num
        self.train_classes_actual = train_classes
        self.test_classes_actual = test_classes

        # Update targets to map to new class indices
        if self.split in ['train', 'val']:
            # In training/validation, convert known classes to their respective indices
            new_targets = []
            for target in self.targets:
                if int(target) in self.train_classes_actual:
                    new_targets.append(self.train_classes_actual.index(int(target)))
                else:
                    # This shouldn't happen in training/val since we filtered to only include train_classes
                    new_targets.append(0)  # Default to first class for safety
            self.targets = new_targets
        else:
            # In testing, convert known classes to their respective indices, unknown to last index
            new_targets = []
            for target in self.targets:
                if int(target) in self.train_classes_actual:
                    new_targets.append(self.train_classes_actual.index(int(target)))
                else:
                    # Map unknown classes to the 'unknown' class index (last index)
                    new_targets.append(self.train_class_num_actual)  # Index of 'unknown' class
            self.targets = new_targets

        self.targets = np.array(self.targets)

        print(f"{self.split.capitalize()} data includes {self.train_class_num_actual + 1} classes "
              f"({self.train_class_num_actual if self.split in ['train', 'val'] else self.test_class_num_actual} original classes), "
              f"{len(self.targets)} samples.")

        if self.split == 'test':
            # Calculate openness for test set
            unique_targets = np.unique(self.targets)
            num_known_in_test = len([t for t in unique_targets if t < self.train_class_num_actual])
            num_unknown_in_test = len([t for t in unique_targets if t == self.train_class_num_actual])
            total_test_classes = num_known_in_test + num_unknown_in_test

            if total_test_classes > 1:  # Need at least 2 classes for openness calculation
                self.openness = float(num_unknown_in_test) / float(total_test_classes)
                print(f"During testing, openness is {self.openness:.4f}.")
            else:
                print("Not enough classes for openness calculation.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.image_paths[index]
        target = int(self.targets[index])

        # Handle different possible image path formats
        full_img_path = os.path.join(self.root, 'DR_grading', 'DR_grading', img_path)
        if not os.path.exists(full_img_path):
            # Try alternate path format (without duplicate DR_grading)
            full_img_path = os.path.join(self.root, 'DR_grading', img_path)

        if not os.path.exists(full_img_path):
            raise FileNotFoundError(f"Image file not found: {full_img_path}")

        img = Image.open(full_img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image_paths)

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}


if __name__ == '__main__':
    # Example usage
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Test the DDR dataset loader
    trainset = DDR(root='../DDR dataset', train=True, transform=transform,
                   train_class_num=3, test_class_num=5, includes_all_train_class=True)
    print(f"Training set: {len(trainset)} samples, classes: {trainset.classes}")

    testset = DDR(root='../DDR dataset', train=False, transform=transform,
                  train_class_num=3, test_class_num=5, includes_all_train_class=True)
    print(f"Testing set: {len(testset)} samples, classes: {testset.classes}")
    if hasattr(testset, 'openness'):
        print(f"Openness: {testset.openness:.4f}")