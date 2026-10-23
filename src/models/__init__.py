"""
Model architectures and neural network definition modules.
"""

from .resnet_gru import SpatiotemporalResNetGRU, load_trained_model

__all__ = ["SpatiotemporalResNetGRU", "load_trained_model"]
