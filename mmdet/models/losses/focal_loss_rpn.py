import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES

@LOSSES.register_module()
class FocalLossRPN(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 loss_weight=1.0,
                 use_sigmoid=True):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(FocalLossRPN, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self,
                inputs,
                targets,
                weights=None,
                avg_factor=None,
                reduction='mean'):
        """Forward function.

        Args:
            Args:
                inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
                targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
                reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.

        Returns:
            torch.Tensor: The calculated loss
        """
        inputs = inputs.view(-1, 1)
        # in mmdet rpn, foreground anchors are labelled with 0, background anchors are labels with 1.
        targets = 1 - targets
        targets = targets.view(-1, 1).float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p.detach() # ours backgroud inverse focal loss. focus on easy background samples and hard foreground samples
        # p_t = p * targets + (1 - p) * (1 - targets) # original focal loss
        # loss = ce_loss * ((1 - p_t) ** self.gamma)
        loss = ce_loss * torch.pow((1 - p_t), self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if weights is not None:
            weights = weights.view(-1, 1).float()

        loss = weight_reduce_loss(
            loss, weights, reduction=reduction, avg_factor=avg_factor)

        loss = self.loss_weight * loss
        return loss
