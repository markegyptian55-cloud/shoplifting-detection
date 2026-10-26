"""
Production Surveillance Video Inference Engine.
Runs real-time inference on arbitrary CCTV surveillance video clips using ONNX Runtime.
Supports the Tri-Model Suite:
1. VideoMAE Vision Transformer (Part 2 Pretrained Foundation Model)
2. Factorized R(2+1)D 3D CNN (Part 1 From-Scratch Model)
3. ResNet18 + Bi-GRU (Part 1 Transfer Learning Model)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import cv2

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def sample_video_frames(video_path: str, target_frames: int = 16, image_size: int = 224) -> np.ndarray:
    """
    Uniformly samples target_frames from a video container and preprocesses them.
    Returns:
        np.ndarray of shape (1, target_frames, 3, image_size, image_size) in float32.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video container: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        frames_list = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_list.append(frame)
        cap.release()
        total_frames = len(frames_list)
        if total_frames == 0:
            raise ValueError(f"Video file contains 0 readable frames: {video_path}")
        frame_indices = np.round(np.linspace(0, total_frames - 1, target_frames)).astype(int)
        raw_frames = [frames_list[idx] for idx in frame_indices]
    else:
        frame_indices = np.round(np.linspace(0, total_frames - 1, target_frames)).astype(int)
        raw_frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret and frame is not None:
                raw_frames.append(frame)
            elif len(raw_frames) > 0:
                raw_frames.append(raw_frames[-1].copy())
            else:
                raw_frames.append(np.zeros((image_size, image_size, 3), dtype=np.uint8))
        cap.release()

    # Preprocessing: BGR -> RGB, Resize, Normalize
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    processed_frames = []
    for f in raw_frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        # HWC -> CHW, normalized [0, 1]
        chw = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
        processed_frames.append(chw)

    clip = np.stack(processed_frames, axis=0) # (T, 3, H, W)
    clip = (clip - imagenet_mean) / imagenet_std
    batch = np.expand_dims(clip, axis=0) # (1, T, 3, H, W)
    return batch.astype(np.float32)


class ShopliftingInferenceEngine:
    """Production Inference Wrapper using ONNX Runtime."""

    def __init__(self, model_path: str, threshold: Optional[float] = None):
        if not HAS_ORT:
            raise ImportError("onnxruntime is required for inference. Run: pip install onnxruntime")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")

        self.model_path = model_path
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        # Auto-detect threshold based on model type if not explicitly specified
        if threshold is not None:
            self.threshold = threshold
        elif "scratch" in model_path.lower():
            self.threshold = 0.64
        elif "resnet" in model_path.lower() or "videomae" in model_path.lower():
            self.threshold = 0.37
        else:
            self.threshold = 0.50

    def predict(self, clip_tensor: np.ndarray) -> Tuple[str, float, float]:
        """
        Runs forward pass on preprocessed clip tensor.
        Returns:
            Tuple of (predicted_label, probability_theft, latency_ms)
        """
        t0 = time.perf_counter()
        raw_outputs = self.session.run(None, {self.input_name: clip_tensor})
        latency_ms = (time.perf_counter() - t0) * 1000.0

        logits = np.squeeze(raw_outputs[0])
        prob_theft = float(1.0 / (1.0 + np.exp(-logits)))
        label = "Shoplifting" if prob_theft >= self.threshold else "Normal Shopping"
        return label, prob_theft, latency_ms

    def predict_video(self, video_path: str) -> dict:
        """Processes video file and returns complete prediction report."""
        clip_tensor = sample_video_frames(video_path)
        label, prob, latency = self.predict(clip_tensor)
        return {
            "video_path": str(video_path),
            "prediction": label,
            "probability_theft": round(prob, 4),
            "decision_threshold": self.threshold,
            "latency_ms": round(latency, 2),
            "model_path": str(self.model_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Real-Time Shoplifting Video Surveillance Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input surveillance video clip (.mp4 / .avi)")
    parser.add_argument("--model", type=str, default="runs/scratch_3dcnn/shoplifting_3dcnn_scratch_int8.onnx",
                        help="Path to ONNX model artifact")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: 0.64 for Scratch 3D CNN, 0.37 for ResNet-GRU)")

    args = parser.parse_args()

    engine = ShopliftingInferenceEngine(args.model, threshold=args.threshold)
    result = engine.predict_video(args.video)

    print("\n" + "=" * 65)
    print("--- SURVEILLANCE VIDEO INFERENCE RESULT ---")
    print("=" * 65)
    print(f"Video Source       : {result['video_path']}")
    print(f"Prediction         : {result['prediction'].upper()}")
    print(f"Theft Probability  : {result['probability_theft'] * 100:.2f}%")
    print(f"Decision Threshold : {result['decision_threshold']:.2f}")
    print(f"Inference Latency  : {result['latency_ms']:.2f} ms")
    print(f"Engine Model       : {result['model_path']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
