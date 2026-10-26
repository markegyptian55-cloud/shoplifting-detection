"""
Shoplifting Detection & Video Action Recognition Package.
Senior-engineered CV Pipeline supporting both Factorized R(2+1)D 3D CNNs and ResNet18 + Bi-GRU spatiotemporal architectures.
"""

from .infer import ShopliftingInferenceEngine

__version__ = "1.0.0"
__author__ = "Mohamed Mostafa Elbasyouni"
__all__ = ["ShopliftingInferenceEngine", "__version__", "__author__"]

