"""
Model architectures and neural network definition modules.
Supports the Tri-Model Suite:
1. VideoMAE Vision Transformer (Part 2 Pretrained Foundation Model)
2. Factorized R(2+1)D 3D CNN (Part 1 From-Scratch Model)
3. ResNet18 + Bi-GRU (Part 1 Transfer Learning Baseline)
"""

from .resnet_gru import SpatiotemporalResNetGRU, load_trained_model
from .scratch_models import Shoplifting3DCNN_Scratch
from .videomae import ShopliftingVideoMAE, build_videomae_classifier, load_trained_videomae

__all__ = [
    "SpatiotemporalResNetGRU",
    "load_trained_model",
    "Shoplifting3DCNN_Scratch",
    "ShopliftingVideoMAE",
    "build_videomae_classifier",
    "load_trained_videomae",
]
