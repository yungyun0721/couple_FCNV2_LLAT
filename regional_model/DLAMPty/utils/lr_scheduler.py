import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    training_steps: int,
    cycles: float = 0.5,
    last_epoch: int = -1,
    schedule_type: str = "cosine",
):
    def cosine_decay(current_step):
        # Warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # decadence
        progress = (current_step - warmup_steps) / max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(cycles) * 2.0 * progress)))

    def constant(current_step):
        return 1.0

    def constant_warmup(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    match schedule_type:
        case "cosine":
            return LambdaLR(optimizer, cosine_decay, last_epoch)
        case "constant":
            return LambdaLR(optimizer, constant, last_epoch)
        case "constant_warmup":
            return LambdaLR(optimizer, constant_warmup, last_epoch)
        case _:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
