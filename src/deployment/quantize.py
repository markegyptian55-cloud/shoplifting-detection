"""
ONNX INT8 Dynamic Quantization & Inference Parity Verification.
Optimizes models for web browser runtimes (onnxruntime-web) and serverless CPUs.
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_dynamic_int8(
    model_fp32_path: str,
    model_int8_path: str,
    weight_type: QuantType = QuantType.QUInt8,
) -> str:
    """
    Applies dynamic 8-bit integer quantization to reduce storage footprint by ~60-70%.
    """
    os.makedirs(os.path.dirname(os.path.abspath(model_int8_path)), exist_ok=True)
    quantize_dynamic(
        model_input=model_fp32_path,
        model_output=model_int8_path,
        weight_type=weight_type,
    )
    return model_int8_path


def verify_inference_parity(
    pytorch_model: nn.Module,
    onnx_path: str,
    dummy_input: Optional[torch.Tensor] = None,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """
    Asserts numerical inference equivalence between PyTorch and ONNX Runtime.
    Returns (passed_parity, max_absolute_difference).
    """
    pytorch_model.eval()
    if dummy_input is None:
        dummy_input = torch.randn(1, 16, 3, 224, 224, dtype=torch.float32)

    with torch.no_grad():
        pt_out = pytorch_model(dummy_input).detach().cpu().numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_inputs = {"video_clip": dummy_input.cpu().numpy()}
    ort_out = session.run(None, onnx_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    passed = max_diff < tolerance
    return passed, max_diff
