import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import DDR
from datasets.acrima import AcrimaGlaucomaDataset
from sklearn.covariance import EmpiricalCovariance

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
            
        score = output[0, class_idx]
        score.backward()
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        
        # Resize using PIL instead of cv2
        cam_img = Image.fromarray(cam)
        cam_img = cam_img.resize((x.shape[3], x.shape[2]), resample=Image.BILINEAR)
        cam = np.array(cam_img)
        
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        return cam

def get_mahalanobis_stats(model, train_loader, device):
    model.eval()
    features = []
    labels = []
    
    print("Computing class statistics for Mahalanobis distance...")
    with torch.no_grad():
        for data, target in train_loader:
            data = data.to(device)
            # Extract features (penultimate)
            x = model.conv1(data)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            f = torch.flatten(x, 1)
            features.append(f.cpu().numpy())
            labels.append(target.numpy())
            
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    class_means = []
    centered_data = []
    for c in range(5):
        mask = labels == c
        c_feats = features[mask]
        mean = c_feats.mean(axis=0)
        class_means.append(mean)
        centered_data.append(c_feats - mean)
        
    cov = EmpiricalCovariance().fit(np.concatenate(centered_data))
    precision = cov.precision_
    
    return class_means, precision

def compute_mahalanobis(feat, class_means, precision):
    dists = []
    for c in range(5):
        centered = feat - class_means[c]
        d = np.sqrt(np.sum(centered @ precision * centered))
        dists.append(d)
    return min(dists)

def show_cam_on_image(img, mask):
    # Use matplotlib colormap instead of cv2
    heatmap = cm.jet(mask)[..., :3] # Get RGB from RGBA
    heatmap = np.float32(heatmap)
    
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='checkpoints/resnet50_full_5class.pth')
    parser.add_argument('--output_dir', type=str, default='outputs/gradcam/')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    print(f"Loading model from {args.model_path}...")
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    
    # GradCAM setup
    grad_cam = GradCAM(model, model.layer4)
    
    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # We need training data for Mahalanobis stats
    ddr_train = DDR(root='DDR dataset', train=True, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
    train_loader = DataLoader(ddr_train, batch_size=32, shuffle=False, num_workers=4)
    
    class_means, precision = get_mahalanobis_stats(model, train_loader, device)
    
    # 1. Process Unknowns (Glaucoma) - Find "Hardest" (Lowest Distance)
    print("\nProcessing Unknowns (Glaucoma)...")
    acrima = AcrimaGlaucomaDataset(root='DDR dataset/Glaucoma/Database/Images', transform=transform)
    acrima_loader = DataLoader(acrima, batch_size=1, shuffle=False)
    
    unknown_scores = []
    
    for i, (img, _) in enumerate(acrima_loader):
        img = img.to(device)
        # Get features
        with torch.no_grad():
            x = model.conv1(img)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            feat = torch.flatten(x, 1).cpu().numpy()[0]
            
        dist = compute_mahalanobis(feat, class_means, precision)
        unknown_scores.append((i, dist))
        
    # Sort by distance (ascending = hardest/most similar to known)
    unknown_scores.sort(key=lambda x: x[1])
    hardest_unknowns = unknown_scores[:args.num_samples]
    
    print(f"Generating Grad-CAM for top {args.num_samples} hardest Glaucoma samples...")
    for rank, (idx, dist) in enumerate(hardest_unknowns):
        img_tensor, _ = acrima[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        img_path = acrima.images[idx]
        img_name = os.path.basename(img_path)
        
        # Generate CAM
        # We want to see why it thinks it's a specific class (the closest one)
        # So we should target the class with the minimum Mahalanobis distance
        # But GradCAM works on logits. So let's target the class with highest logit.
        # Or better: target the class it is *most confused with* (closest Mahalanobis center).
        
        # Find closest class
        dists = []
        # Re-extract feat (inefficient but safe)
        with torch.no_grad():
             x = model.conv1(img_tensor)
             x = model.bn1(x)
             x = model.relu(x)
             x = model.maxpool(x)
             x = model.layer1(x)
             x = model.layer2(x)
             x = model.layer3(x)
             x = model.layer4(x)
             x = model.avgpool(x)
             feat = torch.flatten(x, 1).cpu().numpy()[0]
             
        for c in range(5):
            centered = feat - class_means[c]
            d = np.sqrt(np.sum(centered @ precision * centered))
            dists.append(d)
        closest_class = np.argmin(dists)
        class_names = ['No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        
        # Generate CAM for that closest class
        mask = grad_cam(img_tensor, class_idx=closest_class)
        
        # Prepare image for visualization
        orig_img = np.array(Image.open(img_path).resize((224, 224)).convert('RGB'))
        orig_img = orig_img / 255.0
        vis = show_cam_on_image(orig_img, mask)
        
        save_name = f"unknown_rank{rank+1}_{img_name}_confused_{class_names[closest_class]}.jpg"
        # Use PIL to save instead of cv2
        Image.fromarray(vis).save(os.path.join(args.output_dir, save_name))
        print(f"  Saved {save_name} (Dist: {dist:.2f}, Confused: {class_names[closest_class]})")

    # 2. Process Knowns (DDR) - Find "Hardest" (Misclassified or Low Confidence)
    print("\nProcessing Knowns (DDR)...")
    ddr_test = DDR(root='DDR dataset', train=False, transform=transform, train_class_num=5, test_class_num=5, includes_all_train_class=True)
    ddr_loader = DataLoader(ddr_test, batch_size=1, shuffle=False)
    
    known_scores = [] # (idx, confidence, is_correct)
    
    for i, (img, label) in enumerate(ddr_loader):
        img = img.to(device)
        label = label.item()
        with torch.no_grad():
            output = model(img)
            probs = F.softmax(output, dim=1)
            conf, pred = probs.max(dim=1)
            
        is_correct = (pred.item() == label)
        known_scores.append((i, conf.item(), is_correct, label, pred.item()))
        
    # Sort: Incorrect first, then by lowest confidence
    known_scores.sort(key=lambda x: (x[2], x[1])) 
    hardest_knowns = known_scores[:args.num_samples]
    
    # Helper to get image path from DDR dataset
    def get_ddr_image_path(dataset, index):
        img_path = dataset.image_paths[index]
        # Handle different possible image path formats (copied from DDR.__getitem__)
        full_img_path = os.path.join(dataset.root, 'DR_grading', 'DR_grading', img_path)
        if not os.path.exists(full_img_path):
            full_img_path = os.path.join(dataset.root, 'DR_grading', img_path)
        return full_img_path

    print(f"Generating Grad-CAM for top {args.num_samples} hardest DDR samples...")
    for rank, (idx, conf, is_correct, label, pred) in enumerate(hardest_knowns):
        img_tensor, _ = ddr_test[idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        img_path = get_ddr_image_path(ddr_test, idx)
        img_name = os.path.basename(img_path)
        
        # Target the PREDICTED class (to see why it made the mistake)
        mask = grad_cam(img_tensor, class_idx=pred)
        
        orig_img = np.array(Image.open(img_path).resize((224, 224)).convert('RGB'))
        orig_img = orig_img / 255.0
        vis = show_cam_on_image(orig_img, mask)
        
        status = "WRONG" if not is_correct else "CORRECT"
        save_name = f"known_rank{rank+1}_{status}_{img_name}_pred_{class_names[pred]}_true_{class_names[label]}.jpg"
        # Use PIL to save
        Image.fromarray(vis).save(os.path.join(args.output_dir, save_name))
        print(f"  Saved {save_name} (Conf: {conf:.2f}, Pred: {class_names[pred]}, True: {class_names[label]})")

    print("\nDone! Check outputs/gradcam/ for visualizations.")

if __name__ == "__main__":
    main()
