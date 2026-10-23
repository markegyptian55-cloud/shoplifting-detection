# Shoplifting Detection & Action Recognition in Retail Surveillance

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.19%2B-orange.svg)](https://pytorch.org/vision/)
[![ONNX](https://img.shields.io/badge/ONNX-v1.14%2B-blueviolet.svg)](https://onnx.ai/)
[![Hardware](https://img.shields.io/badge/Compute-2x%20Tesla%20T4%20(29GB)-76b900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end spatiotemporal Computer Vision system engineered for retail shoplifting detection and anomaly action recognition in closed-circuit television (CCTV) surveillance feeds. 

The pipeline integrates a **pretrained ResNet-18** spatial feature extractor with a **2-layer Bidirectional Gated Recurrent Unit (Bi-GRU)**, uniform temporal skip sampling ($T = 16$), class-weighted optimization ($\text{pos\_weight} = 1.64$), and zero-leakage camera group stratification. The finalized model is deployed via **ONNX FP32** and **Dynamic INT8 Quantization** ($61.8\%$ compression), optimized for client-side web browser runtimes (`onnxruntime-web`) and serverless microservices.

---

## 1. System Architecture & Temporal Pipeline

```
                                      SURVEILLANCE VIDEO CLIP (Variable Duration: 3s – 74s)
                                                                 │
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │  Fixed Uniform Skip-Interval Sampling (T = 16)   │
                                        │  indices = round(linspace(0, N - 1, 16))         │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Clip-Consistent Spatial Augmentations (v2)       │
                                        │ Input: (T=16, C=3, H=224, W=224)                 │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Spatial Backbone: Pretrained ResNet-18           │
                                        │ Extract 512-d Frame Descriptors                  │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │ (B, 16, 512)
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Temporal Aggregator: 2-Layer Bidirectional GRU   │
                                        │ hidden_dim = 256, dropout = 0.3                  │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │ (B, 16, 512)
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Sequence Mean Pooling across 16 Time Steps       │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │ (B, 512)
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Binary Classification Head (MLP)                 │
                                        │ Linear(512 -> 128) -> ReLU -> Dropout -> Linear  │
                                        └────────────────────────┬─────────────────────────┘
                                                                 │
                                                                 ▼
                                        ┌──────────────────────────────────────────────────┐
                                        │ Logit Output -> Sigmoid -> Decision Threshold p* │
                                        │ 0: Normal Shopping  |  1: Shoplifting / Theft    │
                                        └──────────────────────────────────────────────────┘
```

---

## 2. Benchmark Performance & Empirical Results

Evaluated on an independent, held-out test partition consisting of **121 surveillance clips from 91 mutually exclusive camera scenes**. The optimal decision threshold ($p^* = 0.45$) was tuned strictly on the validation partition.

| Evaluation Metric | Validation Set ($N = 122$) | Held-Out Test Set ($N = 121$) | Benchmark Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Optimal Threshold ($p^*$)** | `0.45` | `0.45` | N/A | Tuned on Val |
| **Classification Accuracy** | **`91.80%`** | **`92.56%`** | $> 85.0\%$ | **Surpassed** |
| **Precision (Shoplifting)** | **`84.62%`** | **`87.76%`** | $> 80.0\%$ | **Surpassed** |
| **Recall (Shoplifting)** | **`95.65%`** | **`93.48%`** | $> 85.0\%$ | **Surpassed** |
| **Macro F1-Score** | **`0.8980`** | **`0.9053`** | $> 0.850$ | **Surpassed** |
| **F1 95% Bootstrap CI** | N/A | **`[0.8387, 0.9608]`** | Bounded | 1000 Iterations |
| **ROC-AUC Score** | **`0.9631`** | **`0.9755`** | $> 0.900$ | **Surpassed** |
| **PR-AUC Score** | **`0.9350`** | **`0.9570`** | $> 0.850$ | **Surpassed** |

### Confusion Matrix on Held-Out Test Set:
* **True Negatives (Normal correctly identified):** `69 / 75` ($92.0\%$)
* **False Positives (Normal false alarms):** `6 / 75` ($8.0\%$)
* **False Negatives (Missed thefts):** `3 / 46` ($6.5\%$)
* **True Positives (Thefts detected):** `43 / 46` ($93.5\%$)

---

## 3. Evaluation Dashboard

![Evaluation Dashboard](runs/evaluation_dashboard.png)

The benchmark dashboard visualizes:
1. **Confusion Matrix (Test Set):** Clear diagonal dominance with only 3 missed shoplifting events.
2. **ROC Curve ($AUC = 0.9755$):** Robust separation between true positive rate and false positive rate.
3. **Precision-Recall Curve ($PR-AUC = 0.9570$):** High precision maintained across varying operational recalls.
4. **Optimization Curves:** Stable loss convergence across Stage 1 (frozen backbone) and Stage 2 (fine-tuning Layer 4).

---

## 4. The Leakage Audit: Why Random Splitting Fails in CCTV Action Recognition

In retail video datasets, multiple sub-clips are frequently cut from the same master camera recording (e.g., `shop_lifter_105.mp4` and `shop_lifter_105_1.mp4`). 

### The Diagnostic Finding
When split using naive random shuffling, sibling sub-clips end up scattered across train and validation sets:
* **Diagnostic B Audit:** Measured pairwise cosine similarity on background descriptors between validation and train clips.
* **Finding:** **$98.4\%$ of validation clips had an identical nearest neighbor ($S = 1.0000$) in the training set**, enabling a static linear model to achieve an artificial validation score without learning any motion patterns.

### The Remediation: Group-Aware Stratified Splitting
We implemented `GroupAwareStratifiedSplitter` which clusters clips via regex parent ID extraction into **637 unique scene groups**, followed by `StratifiedGroupKFold(n_splits=7)`:
$$\text{Train Groups } \cap \text{ Val Groups } = \emptyset, \quad \text{Train Groups } \cap \text{ Test Groups } = \emptyset$$
This guarantees that **no camera scene, cashier angle, or store interior in the test set was ever seen during training**.

---

## 5. Production ONNX Export & INT8 Dynamic Quantization

To support real-time edge processing and client-side web browser surveillance (`onnxruntime-web` / WebAssembly / WebGPU), the PyTorch checkpoint was exported and quantized:

```
┌──────────────────────────────────────┐
│  PyTorch Native Checkpoint           │
│  best_model.pt (~51.91 MB)           │
└──────────────────┬───────────────────┘
                   │ torch.onnx.export (opset=14, dynamic_axes)
                   ▼
┌──────────────────────────────────────┐
│  ONNX FP32 Model                     │
│  shoplifting_resnet_gru_fp32.onnx    │
│  Size: 51.91 MB | Max Diff: 2.4e-7   │
└──────────────────┬───────────────────┘
                   │ onnxruntime.quantization.quantize_dynamic (QUInt8)
                   ▼
┌──────────────────────────────────────┐
│  ONNX INT8 Quantized Model           │
│  shoplifting_resnet_gru_int8.onnx    │
│  Size: 19.81 MB | Max Diff: 0.0434   │
│  Footprint Reduction: 61.8%          │
└──────────────────────────────────────┘
```

| Deployment Artifact | Precision | Disk Size | Compression | Max Divergence | Best Target Environment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`best_model.pt`** | `FP32` | $51.91\text{ MB}$ | Baseline | Reference | Python / PyTorch Server |
| **`shoplifting_resnet_gru_fp32.onnx`** | `FP32` | $51.91\text{ MB}$ | $1.0\times$ | $0.00000024$ | Cloud GPU / Triton Server |
| **`shoplifting_resnet_gru_int8.onnx`** | **`INT8`** | **`19.81 MB`** | **`61.8% Smaller`** | **$0.0434$** | **Browser (`onnxruntime-web`), WASM, Vercel** |

---

## 6. Repository Layout

```
.
├── requirements.txt                          # Production dependency bounds
├── README.md                                 # Documentation & benchmark report
├── AGENTS.md                                 # Engineering ledger & execution history
├── notebooks/
│   └── shoplifting-detection-pipeline.ipynb  # Executed Kaggle notebook with outputs
├── runs/
│   ├── best_model.pt                         # Trained PyTorch checkpoint (13.6M params)
│   ├── shoplifting_resnet_gru_fp32.onnx      # Standard ONNX FP32 artifact (51.9 MB)
│   ├── shoplifting_resnet_gru_int8.onnx      # Quantized INT8 edge artifact (19.8 MB)
│   ├── evaluation_dashboard.png              # 4-panel benchmark figure
│   └── training_history.json                 # Loss and validation metric history
├── data/
│   └── splits/
│       ├── train_split.csv                   # 612 clips (455 camera groups)
│       ├── val_split.csv                     # 122 clips (91 camera groups)
│       └── test_split.csv                    # 121 clips (91 camera groups)
├── src/
│   ├── __init__.py                           # Package exports
│   ├── data/
│   │   ├── sampling.py                       # Uniform T=16 frame extractor
│   │   ├── transforms.py                     # torchvision.transforms.v2 video pipeline
│   │   ├── dataset.py                        # PyTorch VideoDataset returning (T, C, H, W)
│   │   └── split.py                          # Group-Aware leakage-free split engine
│   ├── models/
│   │   └── resnet_gru.py                     # SpatiotemporalResNetGRU architecture
│   ├── training/
│   │   ├── loss.py                           # BCEWithLogitsLoss (pos_weight = 1.64)
│   │   └── trainer.py                        # Two-stage trainer with BatchNorm eval lock
│   ├── evaluation/
│   │   ├── threshold.py                      # Validation threshold optimizer (p*)
│   │   └── metrics.py                        # Accuracy, Precision, Recall, F1, Bootstrap CI
│   └── deployment/
│       ├── export.py                         # PyTorch to ONNX exporter
│       └── quantize.py                       # INT8 Dynamic Quantization & parity checker
└── tests/
    └── test_pipeline_modules.py              # Offline automated test suite (Zero download)
```

---

## 7. Installation & Quickstart

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* CUDA-compatible GPU (optional; CPU execution is fully supported)

### 1. Clone & Setup Environment
```bash
git clone https://github.com/markegyptian55-cloud/shoplifting-detection.git
cd shoplifting-detection

pip install -r requirements.txt
```

### 2. Run Offline Unit Tests
Run the offline test suite verifying tensor contracts, mathematical bounds, forward passes, and ONNX parity without downloading any data:
```bash
python tests/test_pipeline_modules.py
```

### 3. Run Inference on a Video Clip
```python
import torch
from src.data.sampling import UniformVideoSampler
from src.data.transforms import get_video_transforms
from src.models.resnet_gru import load_trained_model

# 1. Load trained model
model = load_trained_model("runs/best_model.pt", device=torch.device("cpu"))

# 2. Extract 16 uniform frames
sampler = UniformVideoSampler(num_frames=16, target_size=(224, 224))
frames = sampler.decode_and_sample("path/to/cctv_feed.mp4")  # (16, 224, 224, 3)

# 3. Apply clip-consistent transformations
transforms = get_video_transforms(is_train=False)
clip_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (16, 3, 224, 224)
clip_tensor = transforms(clip_tensor).unsqueeze(0)          # (1, 16, 3, 224, 224)

# 4. Predict
with torch.no_grad():
    logit = model(clip_tensor)
    prob = torch.sigmoid(logit).item()

decision_threshold = 0.45
prediction = "Shoplifting" if prob >= decision_threshold else "Normal Shopping"
print(f"Prediction: {prediction} | Confidence: {prob * 100:.2f}%")
```

### 4. Run ONNX Runtime Inference
```python
import onnxruntime as ort
import numpy as np

# Load quantized INT8 model (~19.8 MB)
session = ort.InferenceSession("runs/shoplifting_resnet_gru_int8.onnx", providers=['CPUExecutionProvider'])

# Run forward pass
input_tensor = clip_tensor.numpy()  # (1, 16, 3, 224, 224)
logits = session.run(None, {'video_clip': input_tensor})[0]
prob = 1.0 / (1.0 + np.exp(-logits[0]))
print(f"ONNX Quantized Prediction Probability: {prob:.4f}")
```

---

## 8. License & Author
* **Author:** Mohamed Mostafa Elbasyouni
* **Email:** markegyptian55@gmail.com
* **GitHub:** [@markegyptian55-cloud](https://github.com/markegyptian55-cloud)
* **License:** [MIT License](LICENSE)
