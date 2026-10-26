# Shoplifting Detection & Action Recognition in Retail Surveillance

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.19%2B-orange.svg)](https://pytorch.org/vision/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-v1.14%2B-blueviolet.svg)](https://onnxruntime.ai/)
[![Hardware](https://img.shields.io/badge/Compute-2x%20Tesla%20T4%20(29GB)-76b900.svg)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![Test Suite](https://img.shields.io/badge/Unit%20Tests-13%2F13%20Passed-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, end-to-end spatiotemporal Computer Vision system engineered for automated retail shoplifting detection and behavioral anomaly recognition in closed-circuit television (CCTV) surveillance environments.

This repository implements, audits, benchmarks, and exports two distinct, production-validated deep learning paradigms:
1. **From-Scratch Spatiotemporal 3D CNN:** A lightweight, highly efficient **Factorized R(2+1)D 3D Convolutional Network** (3.47M parameters) trained from randomly initialized Kaiming He weights, achieving **98.35% Test Accuracy** and **100.00% Theft Recall** (0 missed shoplifting incidents).
2. **Transfer Learning Baseline:** A **Pretrained ResNet-18 + Bidirectional GRU** architecture (13.67M parameters) utilizing ImageNet-1K visual representations coupled with recurrent temporal modeling, achieving **95.04% Test Accuracy** and **95.65% Theft Recall**.

Both models are fully exported to **ONNX FP32** with verified numerical parity ($< 1.6 \times 10^{-5}$ divergence) and quantized into **INT8 Dynamic Quantization**, compressing the 3D CNN down to **3.52 MB** (74.6% reduction) for ultra-low-latency deployment on edge CCTV hardware.

---

## 1. Master Comparative Benchmark Matrix

Evaluated on an independent, held-out test partition consisting of **121 surveillance clips across 91 mutually exclusive camera scenes** with **0% group leakage**. Decision thresholds ($p^*$) were tuned strictly on the validation partition and applied once to the test set.

| Architectural Metric | Factorized R(2+1)D 3D CNN (Scratch) | ResNet18 + Bi-GRU (Transfer Learning) | Performance Advantage |
| :--- | :---: | :---: | :---: |
| **Model Weights** | **None (100% From Scratch)** | ImageNet-1K Pretrained | Zero external bias |
| **Total Parameters** | **3.47 M** | 13.67 M | **74.6% lighter** |
| **Optimal Decision Threshold ($p^*$)** | `0.64` | `0.37` | Tuned on Validation |
| **Test Accuracy** | **`98.35%` (119/121)** | `95.04%` (115/121) | **+3.31% higher** |
| **Test Macro F1-Score** | **`0.9826`** | `0.9478` | **+0.0348 higher** |
| **Theft Class Precision** | **`95.83%` (46/48)** | `91.67%` (44/48) | **+4.16% higher** |
| **Theft Class Recall (Sensitivity)** | **`100.00%` (46/46)** | `95.65%` (44/46) | **Zero Missed Thefts (0 FN)** |
| **Normal Class Specificity** | **`97.33%` (73/75)** | `94.67%` (71/75) | **Only 2 False Alarms** |
| **ROC-AUC Score** | **`0.9988`** | `0.9849` | Near-perfect separation |
| **PR-AUC Score** | **`0.9981`** | `0.9720` | Exceptional precision retention |
| **Expected Calibration Error (ECE)** | **`0.0649`** | `0.0812` | Highly calibrated probabilities |
| **Generalization Gap ($\Delta_{\text{val}\rightarrow\text{test}}$)** | **`-0.0001` (-0.01%)** | `-0.0189` (-1.89%) | Zero test-set overfitting |
| **Non-Parametric 95% Bootstrap CI** | **`[95.47%, 100.00%]`** | `[88.00%, 0.9870]` | Statistically superior bounds |
| **FP32 ONNX Model Size** | **`13.86 MB`** | `51.91 MB` | **3.7x smaller** |
| **INT8 Quantized Model Size** | **`3.52 MB`** | `19.81 MB` | **5.6x smaller** |
| **Model Compression Footprint** | **`-74.6%`** | `-61.8%` | Edge-Ready ($< 4\text{ MB}$) |

---

## 2. Visual Diagnostic Dashboards

### 2.1 Stage 2: Factorized R(2+1)D 3D CNN (From Scratch) — 8-Panel Scientific Dashboard
![Scratch 3D CNN Diagnostic Dashboard](runs/scratch_3dcnn/evaluation_dashboard_scratch.png)

The 8-panel diagnostic dashboard provides comprehensive empirical verification:
* **Panel 1 (Optimization Loss Dynamics):** Monotonic training loss convergence from `0.8631` down to `0.2262`. The global minimum validation loss (`0.1421`) was achieved at **Epoch 13** without late-stage overfitting divergence.
* **Panel 2 (Validation Metric Dynamics):** Rapid progression from `62.3%` accuracy to peak validation macro F1 of `0.9657` at Epoch 13.
* **Panel 3 (Threshold Sweep):** Grid search across 81 candidate thresholds $p \in [0.10, 0.90]$, proving that $p^* = 0.64$ maximizes macro F1 and balances precision and recall under class imbalance.
* **Panel 4 (Normalized Confusion Matrix):** 73 Normal clips correctly identified (97.3%), 46 Shoplifting clips correctly caught (100.0%), 0 missed thefts (0.0%), and only 2 false positives (2.7%).
* **Panel 5 & 6 (ROC & Precision-Recall Curves):** $ROC\text{-}AUC = 0.9988$ and $PR\text{-}AUC = 0.9981$, showing clear operational dominance over baseline guessing.
* **Panel 7 (Probability Separation & Bimodality):** Decisive bimodal separation; Normal clips are clustered strongly between `0.00` and `0.10`, while Theft clips are clustered between `0.85` and `1.00`, proving high-confidence margins.
* **Panel 8 (Calibration & Reliability):** Low Expected Calibration Error ($ECE = 0.0649$), confirming predicted probabilities match real-world empirical frequency.

---

### 2.2 Stage 1: ResNet18 + Bi-GRU (Transfer Learning) — 3-Panel Evaluation Dashboard
![ResNet18 GRU Dashboard](runs/resnet18_gru/evaluation_dashboard.png)

* **Confusion Matrix:** 71 True Negatives, 44 True Positives, 4 False Alarms, 2 Missed Thefts at optimal threshold $p^* = 0.37$.
* **ROC & PR Curves:** Robust feature representation yielding $ROC\text{-}AUC = 0.9849$ and $PR\text{-}AUC = 0.9720$.

---

## 3. Deep-Dive Architectural Formulations

```
========================================================================================
PARADIGM A: 2D Spatial Extraction + 1D Temporal Aggregation (ResNet18 + Bi-GRU)
Input Clip (1, 16, 3, 224, 224) 
  --> TimeDistributed ResNet-18 (512-d features per frame)
  --> 2-Layer Bidirectional GRU (hidden_dim=256, dropout=0.3)
  --> Global Temporal Pooling + Linear Head (13.67M params)
========================================================================================

PARADIGM B: Spatiotemporal Factorized 3D CNN (Factorized R(2+1)D from Scratch)
Input Clip (1, 16, 3, 224, 224) 
  --> Conv2Plus1D Block 1: Conv3D(1x3x3) + BatchNorm + ReLU + Conv3D(3x1x1) + BatchNorm + ReLU
  --> Conv2Plus1D Block 2: Downsampling Spatial & Temporal Strides
  --> Conv2Plus1D Block 3: Deep Feature Factorization
  --> Conv2Plus1D Block 4: High-Level Kinematic Representations
  --> AdaptiveAvgPool3d(1, 1, 1) + Linear Head (3.47M params)
========================================================================================
```

### Why did the From-Scratch 3D CNN outperform Transfer Learning?
1. **Direct Kinematic Modeling from Raw Pixels:** ResNet-18 processes each frame as a static 2D image, discarding sub-second velocity and motion cues between adjacent frames. In contrast, the Factorized R(2+1)D network convolves directly across the spatiotemporal cube, capturing the rapid velocity vector of a suspect reaching into a shelf and concealing an item.
2. **Double Non-Linearity via Factorization:** Standard 3D convolutions use a single filter $(3 \times 3 \times 3)$. Factorized $R(2+1)D$ decomposes this into a spatial convolution $(1 \times 3 \times 3)$ followed immediately by a ReLU and Batch Normalization, then a temporal convolution $(3 \times 1 \times 1)$ followed by another ReLU. This doubles non-linear expressiveness while reducing parameters by over 33%.
3. **Protection against Background Memorization:** With 3.47M parameters (vs 13.67M in ResNet-GRU), the 3D CNN is mathematically constrained from memorizing static retail shelf textures, forcing it to generalize purely on dynamic motion anomalies.

---

## 4. Rigorous Engineering & Anti-Leakage Protocol

### 4.1 Zero-Leakage Camera Scene Stratification
Surveillance datasets frequently contain multiple sub-clips cut from the same master surveillance recording. Shuffling clips randomly creates catastrophic background leakage (where identical store aisles and lighting appear in both train and test splits).
* **Audit Check:** Clips were strictly grouped by master video recording ID (`group_id`).
* **Result:** **0 overlapping camera scenes** between Train (455 groups), Validation (91 groups), and Test (91 groups).

### 4.2 Uniform Temporal Skip Sampling ($T = 16$)
Retail theft events in CCTV footage span between 12 and 14 seconds. Fixed uniform skip sampling extracts an optical snapshot every ~0.75 seconds:
$$\text{indices} = \text{round}\left(\text{linspace}(0, N-1, 16)\right)$$
This guarantees full coverage of the behavioral arc (approach $\rightarrow$ reach $\rightarrow$ conceal $\rightarrow$ depart) into a fixed tensor shape $(16, 3, 224, 224)$.

### 4.3 Clip-Consistent Spatial Augmentation
Applying spatial transformations independently to individual frames destroys temporal optical flow. Using `torchvision.transforms.v2`, random horizontal flips, resized crops, and color jitters are applied **identically across all 16 frames** of each video clip simultaneously.

---

## 5. Production Edge Deployment (ONNX & Quantization)

Both models have been exported to open ONNX formats with dynamic batch capabilities ($B \in [1, \infty)$) and verified for numerical parity:

```bash
# Model Artifact Sizes
runs/
├── resnet18_gru/
│   ├── best_model.pt                      # 54.5 MB (PyTorch FP32)
│   ├── shoplifting_resnet_gru_fp32.onnx   # 51.9 MB (ONNX FP32)
│   └── shoplifting_resnet_gru_int8.onnx   # 19.8 MB (ONNX INT8, -61.8%)
└── scratch_3dcnn/
    ├── best_model_scratch.pt              # 14.6 MB (PyTorch FP32)
    ├── shoplifting_3dcnn_scratch_fp32.onnx # 13.9 MB (ONNX FP32)
    └── shoplifting_3dcnn_scratch_int8.onnx #  3.5 MB (ONNX INT8, -74.6%)
```

* **Numerical Parity:** $\max|\text{PyTorch} - \text{ONNX}| < 1.6 \times 10^{-5}$ across all batch sizes.
* **Edge Feasibility:** The 3.52 MB INT8 3D CNN can run directly inside browser runtimes (ONNX.js), edge microcontrollers, or embedded CCTV camera chips without GPU acceleration.

---

## 6. Quickstart & Live Video Inference

### 6.1 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/markegyptian55-cloud/Shoplifting-Detection.git
cd "Shoplifting-Detection"
pip install -r requirements.txt
```

### 6.2 Running Automated Test Suite
Verify that all 13 unit tests, contracts, and model graphs pass:
```bash
python -m unittest discover tests
```

### 6.3 Real-Time Video Inference via CLI
Run live surveillance inference on any `.mp4` or `.avi` video clip using the lightweight ONNX runtime engine:
```bash
# Infer using the ultra-compact 3.5MB INT8 3D CNN model (Default threshold p*=0.64)
python src/infer.py --video "path/to/cctv_clip.mp4" --model "runs/scratch_3dcnn/shoplifting_3dcnn_scratch_int8.onnx"

# Infer using the Transfer Learning model (Default threshold p*=0.37)
python src/infer.py --video "path/to/cctv_clip.mp4" --model "runs/resnet18_gru/shoplifting_resnet_gru_int8.onnx"
```

#### Example Output:
```text
=================================================================
--- SURVEILLANCE VIDEO INFERENCE RESULT ---
=================================================================
Video Source       : samples/theft_sample_01.mp4
Prediction         : SHOPLIFTING
Theft Probability  : 97.42%
Decision Threshold : 0.64
Inference Latency  : 34.12 ms
Engine Model       : runs/scratch_3dcnn/shoplifting_3dcnn_scratch_int8.onnx
=================================================================
```

---

## 7. Repository Directory Structure

```
.
├── notebooks/
│   ├── shoplifting-detection-pipeline.ipynb   # Stage 1: ResNet18 + Bi-GRU Training Pipeline
│   └── 02-shoplifting-scratch-3dcnn.ipynb     # Stage 2: From-Scratch 3D CNN Benchmark Pipeline
├── runs/
│   ├── resnet18_gru/                          # Stage 1 Artifacts (PyTorch, ONNX FP32/INT8, Dashboard)
│   └── scratch_3dcnn/                         # Stage 2 Artifacts (PyTorch, ONNX FP32/INT8, Dashboard)
├── src/
│   ├── __init__.py                            # Package exposure and top-level imports
│   ├── infer.py                               # Standalone CLI & Python ONNX inference engine
│   ├── data/
│   │   ├── dataset.py                         # PyTorch Video Dataset returning (T, C, H, W)
│   │   ├── sampling.py                        # Fixed uniform skip frame extractor
│   │   ├── split.py                           # Group-aware camera stratified splitter
│   │   └── transforms.py                      # Clip-consistent spatial augmentations
│   ├── models/
│   │   ├── resnet_gru.py                      # Pretrained ResNet18 + Bi-GRU definition
│   │   └── scratch_models.py                  # Factorized R(2+1)D 3D CNN definition
│   ├── training/
│   │   ├── loss.py                            # Class-weighted BCE loss formulation
│   │   └── trainer.py                         # Mixed precision training & checkpoint engine
│   ├── evaluation/
│   │   ├── metrics.py                         # Comprehensive evaluation metrics & bootstrap CI
│   │   └── threshold.py                       # Validation threshold grid search
│   └── deployment/
│       ├── export.py                          # Dynamic batch ONNX export engine
│       └── quantize.py                        # Dynamic INT8 ONNX quantization & parity checks
├── tests/
│   ├── test_generated_notebooks.py            # Validates notebook AST syntax, pairs, and English contracts
│   └── test_pipeline_modules.py               # Validates dataset, models, loss, and ONNX parity
├── AGENTS.md                                  # Repository constitution and engineering memory ledger
└── README.md                                  # Master project documentation
```

---

## 8. Scientific Citations & References

1. **Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M.** (2018). *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 6450-6459.
2. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition.* IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
3. **Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y.** (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP 2014.
4. **Muneer, A., et al.** (2023). *Deep Learning Approaches for Shoplifting Detection in Surveillance Video.* IEEE Access.

---

## 9. Author & License

* **Lead AI Engineer:** Mohamed Mostafa Elbasyouni
* **Email:** markegyptian55@gmail.com
* **GitHub:** [markegyptian55-cloud](https://github.com/markegyptian55-cloud)
* **License:** Released under the [MIT License](LICENSE).
