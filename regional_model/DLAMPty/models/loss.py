import torch
import torch.nn as nn


class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        loss = torch.abs(pred - target) * weight

        match self.reduction:
            case 'mean':
                return loss.mean()
            case 'sum':
                return loss.sum()
            case _:
                return loss
