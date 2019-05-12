import torch
import torch.nn as nn
from torch.autograd import Variable

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim() == 4, "expected 4-dimensional data, but instead got {}".format(x.dim())
            horizontal = 2 * x[:,:,1:-1,1:-1] - x[:,:,1:-1,:-2] - x[:,:,1:-1,2:]
            vertical = 2 * x[:,:,1:-1,1:-1] - x[:,:,:-2,1:-1] - x[:,:,2:,1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()
        self.loss = second_derivative(depth)
        return self.loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
