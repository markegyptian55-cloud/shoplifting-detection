"""
PyTorch to ONNX Graph Exporter with Dynamic Batch Dimensions.
"""

import inspect
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import onnx


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int, int, int] = (1, 16, 3, 224, 224),
    opset_version: int = 14,
    device: Optional[torch.device] = None,
) -> str:
    """
    Exports PyTorch Spatiotemporal model to standard ONNX FP32 format.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()

    dummy_clip = torch.randn(*input_shape, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["video_clip"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "video_clip": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    }

    # Ensure backward compatibility across PyTorch 2.0 - 2.6+
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(model, dummy_clip, output_path, **export_kwargs)

    # Validate ONNX graph integrity
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    return output_path
