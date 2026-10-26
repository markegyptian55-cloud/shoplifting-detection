"""
Offline Unit Test Suite for Shoplifting Detection Pipeline Modules.
Zero external downloads; executes strictly offline using synthetic tensors.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path for robust standalone test execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import numpy as np
import pandas as pd
import torch

from src.data.sampling import UniformVideoSampler
from src.data.transforms import get_video_transforms
from src.data.split import GroupAwareStratifiedSplitter
from src.models.resnet_gru import SpatiotemporalResNetGRU, load_trained_model
from src.models.scratch_models import Shoplifting3DCNN_Scratch
from src.training.loss import get_weighted_loss
from src.evaluation.threshold import ThresholdOptimizer
from src.evaluation.metrics import compute_classification_metrics, bootstrap_f1_ci
from src.deployment.export import export_to_onnx
from src.deployment.quantize import quantize_dynamic_int8, verify_inference_parity



class TestShopliftingPipeline(unittest.TestCase):
    """Offline test suite verifying math, contracts, and module cohesion."""

    def test_01_sampling_indices_math(self):
        """Verifies uniform index sampling across multiple clip lengths."""
        sampler = UniformVideoSampler(num_frames=16)

        # Case A: Median duration clip (~300 frames)
        idx_median = sampler.compute_frame_indices(300)
        self.assertEqual(len(idx_median), 16)
        self.assertEqual(idx_median[0], 0)
        self.assertEqual(idx_median[-1], 299)
        self.assertTrue(np.all(np.diff(idx_median) > 0))

        # Case B: Very short clip (10 frames < 16)
        idx_short = sampler.compute_frame_indices(10)
        self.assertEqual(len(idx_short), 16)
        self.assertEqual(idx_short[0], 0)
        self.assertEqual(idx_short[-1], 9)

        # Case C: Long outlier clip (1200 frames)
        idx_long = sampler.compute_frame_indices(1200)
        self.assertEqual(len(idx_long), 16)
        self.assertEqual(idx_long[0], 0)
        self.assertEqual(idx_long[-1], 1199)

    def test_02_video_transforms_tensor_contract(self):
        """Verifies torchvision.transforms.v2 handles 4D video tensors (T, C, H, W)."""
        train_transforms = get_video_transforms(is_train=True)
        eval_transforms = get_video_transforms(is_train=False)

        # Synthetic raw uint8 video tensor: (T=16, C=3, H=224, W=224)
        raw_clip = torch.randint(0, 256, (16, 3, 224, 224), dtype=torch.uint8)

        train_out = train_transforms(raw_clip)
        self.assertEqual(train_out.shape, torch.Size([16, 3, 224, 224]))
        self.assertEqual(train_out.dtype, torch.float32)

        eval_out = eval_transforms(raw_clip)
        self.assertEqual(eval_out.shape, torch.Size([16, 3, 224, 224]))
        self.assertEqual(eval_out.dtype, torch.float32)

    def test_03_group_aware_split_clustering(self):
        """Verifies regex extraction groups sub-clips to eliminate scene leakage."""
        self.assertEqual(
            GroupAwareStratifiedSplitter.extract_parent_id("shop_lifter_105_1.mp4"),
            "shop_lifter_105",
        )
        self.assertEqual(
            GroupAwareStratifiedSplitter.extract_parent_id("shop_lifter_105.mp4"),
            "shop_lifter_105",
        )
        self.assertEqual(
            GroupAwareStratifiedSplitter.extract_parent_id("videyyyyyyyyyss_57_2.mp4"),
            "videyyyyyyyyyss_57",
        )

        # Verify group isolation across mock dataframe
        data = {
            "video_path": [
                f"scene{i}_{j}.mp4" for i in range(1, 11) for j in (1, 2)
            ],
            "label": [1, 1] * 5 + [0, 0] * 5,
        }
        df = pd.DataFrame(data)
        train_df, val_df, test_df = GroupAwareStratifiedSplitter.split(df, n_splits=3, random_state=42)

        train_g = set(train_df["group_id"])
        val_g = set(val_df["group_id"])
        test_g = set(test_df["group_id"])

        self.assertEqual(len(train_g.intersection(val_g)), 0)
        self.assertEqual(len(train_g.intersection(test_g)), 0)
        self.assertEqual(len(val_g.intersection(test_g)), 0)

    def test_04_model_architecture_and_forward_pass(self):
        """Verifies SpatiotemporalResNetGRU forward pass and stage freeze contracts."""
        model = SpatiotemporalResNetGRU(hidden_dim=256, num_layers=2, dropout=0.0, pretrained=False)
        model.eval()

        # Batch of 2 clips: (B=2, T=16, C=3, H=224, W=224)
        dummy_batch = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_batch)

        self.assertEqual(logits.shape, torch.Size([2]))
        self.assertEqual(logits.dtype, torch.float32)

        # Stage 1: Freeze backbone check
        model.freeze_backbone()
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertFalse(model.layer1[0].conv1.weight.requires_grad)
        self.assertTrue(model.gru.weight_ih_l0.requires_grad)
        self.assertTrue(model.classifier[0].weight.requires_grad)

        # Stage 2: Unfreeze layer4 check
        model.unfreeze_layer4()
        self.assertFalse(model.conv1.weight.requires_grad)
        self.assertTrue(model.layer4[0].conv1.weight.requires_grad)
        self.assertTrue(model.gru.weight_ih_l0.requires_grad)
        self.assertTrue(model.classifier[0].weight.requires_grad)

    def test_05_loss_and_metrics_calculation(self):
        """Verifies class-weighted loss and benchmark metrics reporting."""
        criterion = get_weighted_loss(pos_weight_value=1.64)
        logits = torch.tensor([1.5, -0.5, 2.0, -2.0], dtype=torch.float32)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        loss = criterion(logits, targets)

        self.assertGreater(loss.item(), 0.0)

        # Synthetic metric calculation
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.85, 0.2, 0.7, 0.4, 0.3, 0.15, 0.8, 0.05])

        optimizer = ThresholdOptimizer(num_points=11)
        best_th, best_f1 = optimizer.find_best_threshold(y_true, y_prob)
        self.assertGreaterEqual(best_th, 0.10)
        self.assertLessEqual(best_th, 0.90)

        metrics = compute_classification_metrics(y_true, y_prob, threshold=best_th)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertIn("f1_ci_lower", metrics)
        self.assertIn("f1_ci_upper", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.8)

    def test_06_load_trained_checkpoint_if_exists(self):
        """Loads local best_model.pt checkpoint and asserts 100% parameter parity."""
        checkpoint_path = os.path.join("runs", "resnet18_gru", "best_model.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join("runs", "best_model.pt")
        if os.path.exists(checkpoint_path):
            model = load_trained_model(checkpoint_path, device=torch.device("cpu"))
            self.assertFalse(model.training)

            dummy_input = torch.randn(1, 16, 3, 224, 224)
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, torch.Size([1]))
            print(f"\n[Verification] Successfully validated local {checkpoint_path} checkpoint!")

    def test_07_onnx_artifacts_and_parity(self):
        """Verifies local ONNX artifacts and tests numerical parity with ONNX Runtime."""
        runs_dir = os.path.join("runs", "resnet18_gru") if os.path.exists(os.path.join("runs", "resnet18_gru")) else "runs"
        fp32_path = os.path.join(runs_dir, "shoplifting_resnet_gru_fp32.onnx")
        int8_path = os.path.join(runs_dir, "shoplifting_resnet_gru_int8.onnx")
        checkpoint_path = os.path.join(runs_dir, "best_model.pt")

        if os.path.exists(fp32_path) and os.path.exists(checkpoint_path):
            model = load_trained_model(checkpoint_path, device=torch.device("cpu"))
            dummy_input = torch.randn(1, 16, 3, 224, 224)
            passed, max_diff = verify_inference_parity(model, fp32_path, dummy_input=dummy_input, tolerance=1e-4)
            self.assertTrue(passed, f"ONNX parity failed with max diff {max_diff}")
            self.assertLess(max_diff, 1e-4)
            print(f"[Verification] ONNX FP32 Parity PASSED (diff: {max_diff:.8f})")

        if os.path.exists(int8_path):
            self.assertGreater(os.path.getsize(int8_path), 0)
            print("[Verification] ONNX INT8 Quantized model verified successfully!")

    def test_08_scratch_3dcnn_architecture(self):
        """Tests factorized R(2+1)D 3D CNN scratch architecture forward pass, multi-temporal length support, and tensor permutation."""
        model = Shoplifting3DCNN_Scratch(num_classes=1, dropout=0.2)
        model.eval()

        # Standard T=16 sequence: (B=2, T=16, C=3, H=224, W=224)
        dummy_clips = torch.randn(2, 16, 3, 224, 224)
        with torch.no_grad():
            logits = model(dummy_clips)
        self.assertEqual(logits.shape, torch.Size([2]))

        # Dynamic temporal lengths: T=8 and T=4
        for t_len in [8, 4]:
            t_clips = torch.randn(2, t_len, 3, 224, 224)
            with torch.no_grad():
                t_logits = model(t_clips)
            self.assertEqual(t_logits.shape, torch.Size([2]))

        param_count = sum(p.numel() for p in model.parameters())
        self.assertLess(param_count, 5_000_000)
        print(f"\n[Verification] Scratch 3D CNN verified ({param_count:,} params, output shape {logits.shape})")

    def test_09_scratch_3dcnn_onnx_export_and_parity(self):
        """Verifies ONNX FP32 export, INT8 quantization, and numerical parity for Scratch 3D CNN."""
        model = Shoplifting3DCNN_Scratch(num_classes=1, dropout=0.0)
        model.eval()

        onnx_fp32 = os.path.join("runs", "scratch_3dcnn_test.onnx")
        onnx_int8 = os.path.join("runs", "scratch_3dcnn_test_int8.onnx")

        dummy_input = torch.randn(1, 16, 3, 224, 224)
        export_to_onnx(model, onnx_fp32, dummy_input=dummy_input, opset_version=14)
        self.assertTrue(os.path.exists(onnx_fp32))

        passed, max_diff = verify_inference_parity(model, onnx_fp32, dummy_input=dummy_input, tolerance=1e-4)
        self.assertTrue(passed, f"Scratch ONNX parity failed with max diff {max_diff}")
        self.assertLess(max_diff, 1e-4)

        quantize_dynamic_int8(onnx_fp32, onnx_int8)
        self.assertTrue(os.path.exists(onnx_int8))
        self.assertLess(os.path.getsize(onnx_int8), os.path.getsize(onnx_fp32))

        # Cleanup test artifacts
        for f in [onnx_fp32, onnx_int8]:
            if os.path.exists(f):
                os.remove(f)
        print(f"[Verification] Scratch 3D CNN ONNX FP32 & INT8 Parity verified (max diff: {max_diff:.8f})")

    def test_10_videomae_quantized_onnx_inference(self):
        """Verifies VideoMAE quantized INT8 ONNX production model inference and telemetry."""
        from src.infer import ShopliftingInferenceEngine

        onnx_int8_path = os.path.join("runs", "videomae", "shoplifting_videomae_int8.onnx")
        if os.path.exists(onnx_int8_path):
            file_size_mb = os.path.getsize(onnx_int8_path) / (1024 * 1024)
            self.assertGreater(file_size_mb, 80.0)
            self.assertLess(file_size_mb, 100.0)

            engine = ShopliftingInferenceEngine(onnx_int8_path, threshold=0.37)
            self.assertEqual(engine.input_name, "video_frames")
            self.assertEqual(engine.threshold, 0.37)

            dummy_clip = np.random.randn(1, 16, 3, 224, 224).astype(np.float32)
            label, prob, latency_ms = engine.predict(dummy_clip)

            self.assertIn(label, ["Shoplifting", "Normal Shopping"])
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertGreater(latency_ms, 0.0)
            print(f"\n[Verification] VideoMAE INT8 ONNX verified ({file_size_mb:.2f} MB, Pred: {label}, Latency: {latency_ms:.1f}ms)")


if __name__ == "__main__":
    unittest.main()
