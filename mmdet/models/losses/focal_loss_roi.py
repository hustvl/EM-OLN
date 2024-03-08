import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import weight_reduce_loss
from ..builder import LOSSES

@LOSSES.register_module()
class FocalLossROI(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.25,
                 num_classes=20,
                 loss_weight=1.0):
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
        super(FocalLossROI, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = torch.zeros(num_classes+1)
        self.alpha[:num_classes] = 1 - alpha
        self.alpha[num_classes] = alpha # 对背景类进行抑制
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
        # print(f'inputs shape {inputs.shape}')
        # print(f'targets shape {targets.shape}')
        # print(f'tagers max {targets.max()}')
        # if inputs.size(0)<1024:
        #     import pdb;pdb.set_trace()
        inputs = inputs.view(-1, inputs.size(-1))
        inputs_logsoft = F.log_softmax(inputs, 1)
        inputs_softmax = torch.exp(inputs_logsoft)

        # in mmdet rpn, foreground anchors are labelled with [0, N-1], background anchors are labels with N.
        # targets = targets.view(-1, 1)
        inputs_softmax = inputs_softmax.gather(1, targets.view(-1, 1))
        inputs_softmax[targets==self.num_classes] = 1 - inputs_softmax[targets==self.num_classes]
        inputs_softmax = inputs_softmax.detach()
        inputs_logsoft = inputs_logsoft.gather(1, targets.view(-1, 1))

        loss = -torch.mul(torch.pow((1-inputs_softmax), self.gamma), inputs_logsoft)

        self.alpha = self.alpha.to(inputs.device)
        self.alpha = self.alpha.gather(0, targets)
        loss = torch.mul(self.alpha, loss.squeeze())

        if weights is not None:
            weights = weights.view(-1, 1).float()
        # import pdb;pdb.set_trace()
        loss = weight_reduce_loss(
            loss, weights.squeeze(), reduction=reduction, avg_factor=avg_factor)

        loss = self.loss_weight * loss
        # print(f'focal loss {loss}')
        return loss
