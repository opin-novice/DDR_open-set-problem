from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_url, check_integrity


class DDR(data.Dataset):
    """DDR Diabetic Retinopathy Detection Dataset

    Args:
        root (string): Root directory of dataset where ``DDR/processed/training.pt``
            and  ``DDR/processed/test.pt`` exist.
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

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, train_class_num=3, test_class_num=5, includes_all_train_class=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_class_num = train_class_num
        self.test_class_num = test_class_num
        self.includes_all_train_class = includes_all_train_class

        # Load the CSV file containing the image paths and labels
        csv_path = os.path.join(self.root, 'DR_grading.csv')
        if not os.path.exists(csv_path):
            raise RuntimeError('DDR dataset CSV file not found. You can use download=True to download it '
                               'or put it in the root directory.')

        dataframe = pd.read_csv(csv_path)
        self.image_paths = dataframe.iloc[:, 0].values  # First column contains image IDs
        self.targets = dataframe.iloc[:, 1].values     # Second column contains labels

        # Convert targets to integers
        self.targets = self.targets.astype(np.int64)

        # Define classes (0 to 4 for DDR - representing different stages of diabetic retinopathy)
        self.classes = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative_DR']

        # Apply open set transformation
        self._update_open_set()

    def _update_open_set(self):
        """
        Update the dataset to simulate open set recognition scenario
        """
        print(f"Original classes: {self.classes}")
        print(f"Training with {self.train_class_num} known classes, Testing with {self.test_class_num} total classes")

        # Define class list (0 to 4 for DDR dataset)
        class_list = list(range(len(self.classes)))

        # Select training classes (typically 0 to train_class_num-1)
        train_classes = list(range(self.train_class_num))

        # For testing, select test_class_num classes
        if self.includes_all_train_class:
            # Include all training classes plus additional unknown classes
            if self.train:
                test_classes = train_classes
            else:
                test_classes = train_classes + list(range(self.train_class_num, self.test_class_num))
        else:
            # Randomly select classes
            if self.train:
                test_classes = train_classes
            else:
                test_classes = list(range(self.test_class_num))

        # Update classes to include 'unknown' label
        selected_elements = [self.classes[index] for index in train_classes]
        selected_elements.append('unknown')
        self.classes = selected_elements

        print(f"Updated classes: {self.classes}")

        # Create boolean mask for selecting samples based on train/test split
        if self.train:
            # Training phase: only use samples from training classes
            indexes = [i for i, x in enumerate(self.targets) if x in train_classes]
        else:
            # Testing phase: use samples from test classes
            indexes = [i for i, x in enumerate(self.targets) if x in test_classes]

        # Update the image paths and targets based on selected indexes
        self.image_paths = [self.image_paths[i] for i in indexes]
        self.targets = [self.targets[i] for i in indexes]

        # Update targets to map to new class indices
        if self.train:
            # In training, convert known classes to their respective indices
            new_targets = []
            for target in self.targets:
                if int(target) in train_classes:
                    new_targets.append(train_classes.index(int(target)))
                else:
                    # This shouldn't happen in training since we filtered to only include train_classes
                    new_targets.append(0)  # Default to first class for safety
            self.targets = new_targets
        else:
            # In testing, convert known classes to their respective indices, unknown to last index
            new_targets = []
            for target in self.targets:
                if int(target) in train_classes:
                    new_targets.append(train_classes.index(int(target)))
                else:
                    # Map unknown classes to the 'unknown' class index (last index)
                    new_targets.append(self.train_class_num)  # Index of 'unknown' class
            self.targets = new_targets

        self.targets = np.array(self.targets)

        print(f"{'Training' if self.train else 'Testing'} data includes {self.train_class_num + 1} classes "
              f"({self.train_class_num if self.train else self.test_class_num} original classes), "
              f"{len(self.targets)} samples.")

        if not self.train:
            # Calculate openness for test set
            unique_targets = np.unique(self.targets)
            num_known_in_test = len([t for t in unique_targets if t < self.train_class_num])
            num_unknown_in_test = len([t for t in unique_targets if t == self.train_class_num])
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
        full_img_path = os.path.join(self.root, 'DR_grading', 'DR_grading', str(img_path) + '.jpg')
        if not os.path.exists(full_img_path):
            # Try alternate path format (without duplicate DR_grading)
            full_img_path = os.path.join(self.root, 'DR_grading', str(img_path) + '.jpg')

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

    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

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