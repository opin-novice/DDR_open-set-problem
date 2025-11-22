import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob

class AcrimaGlaucomaDataset(Dataset):
    """
    ACRIMA Dataset for OOD Detection (Glaucoma as Unknown)
    
    We only load the Glaucoma images to serve as "Unknown" samples.
    Path: DDR dataset/Glaucoma/Database/Images
    Glaucoma images have '_g_' in their filename.
    """
    def __init__(self, root='DDR dataset/Glaucoma/Database/Images', transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Verify directory exists
        if not os.path.exists(root):
            raise ValueError(f"Directory not found: {root}")
            
        # Find all images
        all_files = glob.glob(os.path.join(root, "*_ACRIMA.jpg")) + glob.glob(os.path.join(root, "*_ACRIMA.JPG"))
        
        # Filter for Glaucoma images (filenames containing '_g_')
        for file_path in all_files:
            filename = os.path.basename(file_path)
            if '_g_' in filename:
                self.images.append(file_path)
                self.labels.append(5) # Label 5 for Unknown (since DDR has 0-4)
                
        print(f"Loaded {len(self.images)} Glaucoma images from ACRIMA")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback to avoid crashing
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
