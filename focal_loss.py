"""
Focal Loss Implementation for Class Imbalance
Prevents minority class collapse (Class 1 = Mild DR)

Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
- gamma: focusing parameter (higher = more focus on hard examples)
- alpha: class weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for addressing class imbalance
        
        Args:
            alpha: Class weights (list or tensor), e.g., [0.4, 1.0, 0.6]
            gamma: Focusing parameter, default 2.0
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.FloatTensor(alpha)
            self.alpha = alpha
        else:
            self.alpha = None
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: predicted logits (N, C)
            targets: ground truth labels (N,)
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Focal loss formula
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss
    Alternative to Focal Loss for extreme imbalance
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / torch.sqrt(torch.sqrt(torch.FloatTensor(cls_num_list)))
        m_list = m_list * (max_m / m_list.max())
        self.m_list = m_list
        self.s = s
        
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        if self.m_list.device != x.device:
            self.m_list = self.m_list.to(x.device)
        
        index_float = index.type(torch.FloatTensor).to(x.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
        
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target)
