import numpy
import torch

__all__ = ["WeatherData", "WeatherDataNumpy", "Shape_T", "Stat_T"]

WeatherData = tuple[torch.Tensor, torch.Tensor]
WeatherDataNumpy = tuple[numpy.ndarray, numpy.ndarray]
Shape_T = tuple[int, int, int]
Stat_T = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
