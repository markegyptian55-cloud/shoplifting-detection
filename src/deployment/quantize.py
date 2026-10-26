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


def _apply_windows_onnx_quant_fix() -> None:
    """
    Guards against onnx 1.22+ external data canonicalization 'Access is denied' bug
    during onnxruntime save_and_reload_model_with_shape_infer on Windows.
    """
    try:
        import tempfile
        from pathlib import Path
        import onnx
        import onnxruntime.quantization.quant_utils as qu
        import onnxruntime.quantization.onnx_quantizer as oq

        def _safe_save_and_reload(model):
            with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
                model_path = Path(quant_tmp_dir) / "model.onnx"
                inferred_path = Path(quant_tmp_dir) / "model-inferred.onnx"
                onnx.save_model(model, str(model_path))
                onnx.shape_inference.infer_shapes_path(str(model_path), str(inferred_path))
                inferred = onnx.load(str(inferred_path))
                qu.add_infer_metadata(inferred)
                return inferred

        qu.save_and_reload_model_with_shape_infer = _safe_save_and_reload
        oq.save_and_reload_model_with_shape_infer = _safe_save_and_reload
    except Exception:
        pass


def quantize_dynamic_int8(
    model_fp32_path: str,
    model_int8_path: str,
    weight_type: QuantType = QuantType.QUInt8,
    per_channel: bool = False,
) -> str:
    """
    Applies dynamic 8-bit integer quantization to reduce storage footprint by ~60-75%.
    """
    _apply_windows_onnx_quant_fix()
    os.makedirs(os.path.dirname(os.path.abspath(model_int8_path)), exist_ok=True)
    quantize_kwargs = {
        "model_input": model_fp32_path,
        "model_output": model_int8_path,
        "weight_type": weight_type,
    }
    if per_channel:
        quantize_kwargs["per_channel"] = True

    quantize_dynamic(**quantize_kwargs)
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
    input_name = session.get_inputs()[0].name
    onnx_inputs = {input_name: dummy_input.cpu().numpy()}
    ort_out = session.run(None, onnx_inputs)[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    passed = max_diff < tolerance
    return passed, max_diff
