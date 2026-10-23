"""
Deployment utilities: ONNX export, INT8 Dynamic Quantization, and numerical parity checks.
"""

from .export import export_to_onnx
from .quantize import quantize_dynamic_int8, verify_inference_parity

__all__ = [
    "export_to_onnx",
    "quantize_dynamic_int8",
    "verify_inference_parity",
]
