import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection and imbalanced classification.
    
    Args:
        alpha (float or list): Weighting factor for each class. If float, it's applied to the rare class.
                               If list, it should match the number of classes.
        gamma (float): Focusing parameter.
        reduction (str): 'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions of shape (batch_size, num_classes)
            targets: Ground truth labels of shape (batch_size)
        """
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedLoss(nn.Module):
    """
    Class-balanced loss that automatically calculates weights based on class counts.
    """
    def __init__(self, class_counts, gamma=2.0, beta=0.9999):
        super(BalancedLoss, self).__init__()
        self.class_counts = torch.tensor(class_counts).float()
        self.num_classes = len(class_counts)
        self.beta = beta
        self.gamma = gamma
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, self.class_counts)
        weights = (1.0 - self.beta) / effective_num
        self.weights = weights / weights.sum() * self.num_classes
        
        self.focal_loss = FocalLoss(alpha=self.weights.tolist(), gamma=gamma)

    def forward(self, inputs, targets):
        return self.focal_loss(inputs, targets)
