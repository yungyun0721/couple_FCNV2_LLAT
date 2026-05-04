from dataclasses import dataclass

import torch


@dataclass
class RMSEAggregator:
    rmse_mean: torch.Tensor
    rmse_count: int

    def update(self, rmse: torch.Tensor) -> None:
        self.rmse_mean = (self.rmse_mean * self.rmse_count + rmse) / (self.rmse_count + 1)
        self.rmse_count += 1
