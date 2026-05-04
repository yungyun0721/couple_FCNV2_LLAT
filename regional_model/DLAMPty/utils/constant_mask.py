import numpy as np
import torch


def LoadConstantMask(path: str) -> torch.Tensor:
    """
    Load the constant mask from the given path.

    Args:
        path (str): Path to the constant mask.

    Returns:
        torch.Tensor: Constant mask of shape (latitude, longitude).
    """
    mask = np.load(path)
    mask = torch.from_numpy(mask)
    return mask

def LoadDummyMask(value:float,shape:tuple) -> torch.Tensor:
    """
    Load the constant mask from the given path.

    Args:
        path (str): Path to the constant mask.

    Returns:
        torch.Tensor: Constant mask of shape (latitude, longitude).
    """
    mask = np.zeros(shape)+value
    mask = torch.from_numpy(np.zeros(shape))
    return mask
