import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import random

class CLAHE(object):
    """Apply Contrast Limited Adaptive Histogram Equalization to the image."""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: CLAHE applied image.
        """
        # Convert to numpy array
        img_np = np.array(img)
        
        # Check if image is RGB or grayscale
        if len(img_np.shape) == 3:
            # Convert to LAB color space
            import cv2
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            lab_planes[0] = clahe.apply(lab_planes[0])
            
            # Merge channels
            lab = cv2.merge(lab_planes)
            
            # Convert back to RGB
            img_with_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_with_clahe)
        else:
            # Grayscale
            import cv2
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_with_clahe = clahe.apply(img_np)
            return Image.fromarray(img_with_clahe)

    def __repr__(self):
        return self.__class__.__name__ + '(clip_limit={0}, tile_grid_size={1})'.format(
            self.clip_limit, self.tile_grid_size)

class RandomRotate90(object):
    """Randomly rotate the image by 0, 90, 180, or 270 degrees."""
    def __call__(self, img):
        angle = random.choice([0, 90, 180, 270])
        return img.rotate(angle)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class GaussianBlur(object):
    """Apply Gaussian Blur to the image."""
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(self.radius))

    def __repr__(self):
        return self.__class__.__name__ + '(radius={0})'.format(self.radius)

def get_ddr_transforms(input_size=224, is_training=True):
    """
    Get transforms for DDR dataset.
    
    Args:
        input_size (int): Input image size.
        is_training (bool): Whether to return training or testing transforms.
        
    Returns:
        torchvision.transforms.Compose: Composed transforms.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RandomRotate90(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            # CLAHE disabled due to Numpy 2.x conflict with OpenCV
            # Using Torchvision alternatives for contrast enhancement
            transforms.RandomEqualize(p=0.2),
            transforms.RandomAutocontrast(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
