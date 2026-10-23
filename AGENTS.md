# AGENTS.md — Project Operating Principles & Engineering Constitution
**Project:** Shoplifting Detection (Cellula Technologies — Week 5 / Part 1)  
**Workspace:** `F:\projects\Shoplifting Detection`  
**Target Domain:** Temporal Video Classification & Action Recognition for Security Surveillance  

---

## 1. Prime Directives & Binding Golden Rules (Strictly Enforced)

Any agent operating in this repository must strictly obey these foundational rules without exception:

### 1.1 Zero Automatic Execution & Zero Silent Changes
* **Role Definition:** The Agent acts as an expert Computer Vision / Deep Learning researcher, software architect, and educator.
* **Explicit Authorization Required:** The Agent is strictly forbidden from writing or modifying code in project files or Jupyter notebooks until the user provides clear, explicit verbal approval (e.g., `"نفذ الجزء دا"`, `"توكل على الله ونفذ"`, or equivalent).
* **Plan First Protocol:** For any non-trivial phase, the Agent must first present a clear, structured implementation plan detailing mathematical formulation, code design, risks, and verification steps.

### 1.2 Absolute Ban on Automatic `git push`
* **Under NO circumstance may the Agent run `git push` automatically.**
* Local staging and commits (`git add`, `git commit`) are allowed when requested or approved.
* `git push` is exclusively permitted upon a direct, standalone instruction from the user.

### 1.3 Strict Language Isolation
* **In Conversation / Chat:** Use natural, clear Egyptian Arabic for explanations, physical/mathematical reasoning, and collaborative discussions.
* **Inside All Project Files (Notebooks, Scripts, Docs, Markdown cells, Code comments):** **100% English Only.** Never write a single Arabic word inside a notebook, python script, config, commit message, or diagram.
* **Anti-BiDi Formatting in Chat:** When explaining in Arabic, never mix English and Arabic within the same sentence or inside parentheses. Always isolate English terms, filenames, code snippets, and IDs inside backticks (`` `term` ``), on separate lines, or inside Markdown tables.

### 1.4 Human-Style Senior Engineering (Anti-AI-Bloat)
* Write concise, expressive, human-like Python code.
* Avoid robotic, verbose AI boilerplate, meaningless wrapper functions, and dummy fallback stubs.
* **Minimise Code Cells:** Combine logically cohesive steps into fewer, unified, purposeful cells rather than fragmenting the workflow into dozens of trivial micro-cells.

### 1.5 Surgical Modifications & Evidence-Backed Metrics
* Never perform wholesale rewrites of working files. Apply minimal, surgical diffs.
* **Zero Invented Numbers:** Every metric, accuracy score, parameter count, duration, and latency reported in documentation must originate from actual local or Kaggle/Colab execution logs.

### 1.6 User Interaction & Cell-by-Cell Delivery Protocol
* **Non-Expert Friendly Interaction:** Never ask the user to manually edit partial snippets or hunt down lines of code.
* **Full Replacement Code Blocks:** Always provide the complete, copy-pasteable replacement code for the target cell in a single code block.
* **Strict Cell-by-Cell Flow:** Deliver exactly one cell pair at a time (Markdown Cell followed by Code Cell). Wait for the user to execute it on Kaggle and report output before proceeding. If errors occur, diagnose and provide the full corrected cell block.
* **Short, Direct, Human Python:** Keep code concise, well-documented, clean, and free of over-engineered boilerplate.

### 1.7 Cell-Level Independence & Self-Healing Resilience
* **Zero Fragile Dependencies:** Every code cell must be defensive, self-contained, and executable independently without requiring the user to run all preceding cells from the top.
* **Kernel Restart Resilience:** If a session disconnects, restarts, or clears its memory, running any cell must never crash with `NameError` on missing variables.
* **Auto-Resolution Contract:** Each cell must import its own required modules, automatically resolve `DATASET_ROOT` and `device` if missing from global memory, and load cached artifacts/splits from `/kaggle/working/` or re-instantiate them automatically.

### 1.8 Full Hardware Maximization & Dual-GPU Exploitation
* **Active Hardware Profile:** 2x NVIDIA Tesla T4 (29.1 GB Total VRAM).
* **Zero Idle GPU Quota:** Since GPU session time is active and metered on Kaggle, all computational modeling, tensor transformations, and training/evaluation MUST aggressively exploit the GPU resources.
* **Dual-GPU Scaling:** Wrap models in `torch.nn.DataParallel` across both GPUs (`device_ids=[0, 1]`) during training and evaluation to process larger batch sizes ($B = 32$) and double throughput.
* **Mixed Precision Acceleration:** Enforce `torch.cuda.amp.autocast(dtype=torch.float16)` to engage NVIDIA Tensor Cores for ~2x faster matrix operations and minimal memory overhead.
* **CUDA-First Pipeline:** Transfer tensors and run transforms directly on CUDA where supported to keep the GPU pipeline saturated.

---

## 2. Mandatory Notebook Architecture & Markdown Numbering Standard

To guarantee top-tier academic and industrial presentation, all Jupyter Notebooks in this project must follow a strict, numbered hierarchy.

### 2.1 The Two-Cell Rhythm (Markdown Specification $\rightarrow$ Focused Code Cell)
Every computational step in the notebook must strictly follow a **two-cell pair**:
1. **Markdown Documentation Cell:** Explicit numbered heading, context, mathematical explanation, and input/output contracts.
2. **Focused Implementation Code Cell:** A single, clean, cohesive block executing the documented step without artificial fragmentation.

```
┌────────────────────────────────────────────────────────────────────────┐
│ [Markdown Cell] #.X Formal Title + Context + Expected Contract         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ [Code Cell] Clean, robust, human-engineered Python implementation      │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Standard Numbering Schema
Notebooks must strictly adhere to this logical numbering hierarchy:

* `0.0 Executive Overview & Scientific Objectives`
* `1.0 Environment Setup & Hardware Telemetry`
  * `1.1 Dependency Verification & Deterministic Seed Configuration`
* `2.0 Video Dataset Discovery & Exploratory Data Analysis (EDA)`
  * `2.1 Class Distribution & Imbalance Audit`
  * `2.2 Clip Duration, Frame Rate (FPS) & Resolution Distribution`
  * `2.3 Video Integrity & Corrupted Container Verification`
* `3.0 Temporal Frame Sampling & Augmentation Engine`
  * `3.1 Uniform Skip Interval Frame Sampling (Target T Frames)`
  * `3.2 Clip-Consistent Spatial Augmentation (Joint Flip, Crop, Jitter)`
  * `3.3 PyTorch Video Dataset Pipeline (Returning Tensor [T, C, H, W])`
  * `3.4 Leakage-Free Stratified Train / Val / Test Split`
* `4.0 Model Architecture Design & Implementation`
  * `4.1 Temporal Model Definition (3D CNN or 2D CNN-Backbone + RNN/GRU)`
  * `4.2 Model Parameter Footprint & Forward Pass Tensor Sanity Check`
* `5.0 Training, Validation & Optimization Loop`
  * `5.1 Loss Function Formulation & Class-Weighted Handling`
  * `5.2 Mixed Precision (AMP) & Cosine Annealing Learning Rate Scheduler`
  * `5.3 Model Checkpointing & Early Stopping on Validation F1/Loss`
* `6.0 Rigorous Evaluation & Benchmark Analytics`
  * `6.1 Held-Out Test Set Evaluation (Accuracy, Precision, Recall, F1)`
  * `6.2 Confusion Matrix & ROC-AUC Analysis`
  * `6.3 Failure Mode Analysis & Hard Negative Inspection`
* `7.0 Model Export & Deployment Readiness`

### 2.3 Exact Markdown Cell Design Template
Each Markdown cell must provide clear technical value:

```markdown
### 3.1 Uniform Skip Interval Frame Sampling

**Objective:**  
Sample a fixed sequence of $T$ frames from any arbitrary video clip of length $N$ frames such that the temporal coverage spans the entire duration uniformly.

**Mathematical Formulation:**  
Given total frames $N$ and target frames $T$:
$$\text{indices} = \left\lfloor i \times \frac{N}{T} \right\rfloor \quad \text{for } i \in \{0, 1, \dots, T - 1\}$$
For short clips where $N < T$, edge frames are padded or looped to maintain a fixed temporal horizon.

**Expected Tensor Output Contract:**  
- Input: Video path (`.mp4`, `.avi`)
- Output: List of $T$ PIL Images or NumPy array of shape $(T, H, W, C)$ with $C=3$.
```

Followed immediately by the unified Python implementation cell.

---

## 3. Computer Vision & Temporal Video Engineering Specifics

When tackling video-based action recognition and shoplifting detection:

### 3.1 Data Contracts & Tensor Geometry
* **Clip Tensor Contract:** PyTorch video models expect 5D tensors:
  * For 3D CNNs: `(B, C, T, H, W)` or `(B, T, C, H, W)` depending on architecture (e.g., PyTorch `torchvision.models.video` uses `(B, C, T, H, W)`).
  * For CNN + RNN/Transformer: `(B, T, C, H, W)` where $B$ is batch size, $T$ is temporal frames, $C$ is color channels (3), $H \times W$ is spatial resolution (e.g. $224 \times 224$ or $112 \times 112$).
* **Clip-Consistent Spatial Augmentations:**
  * Spatial transformations (RandomHorizontalFlip, RandomCrop, ColorJitter) **must be identical across all $T$ frames** of a single clip. Independent frame augmentation destroys temporal visual continuity.

### 3.2 Data Leakage & Split Strategy
* **Zero Video Leakage:** If raw datasets contain multiple sub-clips extracted from the same master surveillance recording, all sub-clips from that recording must remain strictly inside the same split (grouped split).
* **Stratification:** Splits must preserve class balance (Normal Shopping vs Shoplifting).

### 3.3 Evaluation Metrics for Anomaly / Security Tasks
* Raw Accuracy is insufficient due to class imbalance in real-world retail surveillance.
* **Primary Decision Metric:** **Macro/Binary F1-Score**, Precision, Recall, and PR-AUC.

---

## 4. Agent Memory & Evolutionary Roadmap System

Any Agent working in this repository must maintain and update this section as milestones are achieved:

### 4.1 Dataset & Environmental Ledger
* **Platform:** Kaggle Notebook (`GPU T4 x2`, 29 GB total VRAM, Internet enabled)
* **Dataset Root:** `/kaggle/input/datasets/mohamedelbasyouni55/shoplifting-surveillance-dataset/shoplifting_dataset/Shop DataSet`
* **Verified Volume & Classes:**
  * Total Clips: 855 video files
  * Normal Shopping (`non shop lifters` / class 0): 531 clips (62.1%)
  * Shoplifting / Theft (`shop lifters` / class 1): 324 clips (37.9%)
  * Class Ratio: 1.64 : 1 (Mild imbalance, handled by `pos_weight=1.64`)
* **Spatial Resolution:** 100% uniform at `704x576` (PAL standard retail CCTV, zero spatial outliers)
* **Clip Durations:**
  * Normal Shopping: Min 3.0s, Median 14.0s, Max 74.0s (Mean frames: 347.78)
  * Shoplifting: Min 4.0s, Median 12.0s, Max 23.0s (Mean frames: 304.60)
  * Duration Outlier: 1 clip at 74.0s (effectively normalized by fixed $T=16$ uniform sampling)
* **Sampling Parameters:** **Fixed, uniform $T = 16$ frames** across entire duration for train, val, and test.
* **Frame Caching:** Decode each video ONCE, resize to $224 \times 224$, and cache the 16 uniform frames to disk.
* **Architecture:** ResNet18 (ImageNet weights, feature extractor) + GRU temporal aggregator + Linear head.
* **Hardware Acceleration Engine:** Dual-GPU `torch.nn.DataParallel` over `[GPU 0, GPU 1]` with `torch.cuda.amp.autocast` (Mixed Precision FP16) and batch size $B=32$ (16 per GPU).
* **Training Protocol:** Two-stage training:
  * Stage 1: Frozen ResNet18 backbone, train GRU + head only (`AdamW`, `lr ~ 1e-3`, `pos_weight = 1.64`).
  * Stage 2: Unfreeze `layer4` of ResNet18 with lower `lr` (`~ 1e-4` or `5e-5`), keeping backbone batch norm in eval mode.
* **Validation & Threshold:** Tune classification threshold on validation set for optimal F1-score; evaluate once on held-out test set.
* **Data Split Strategy:** 70% Train / 15% Val / 15% Test grouped by source video / scene ID to eliminate leakage.

### 4.2 Research Findings & Literature Ledger

| # | Topic / Reference | Finding & Source Evidence | Bound Decision |
|---|---|---|---|
| 1 | Architecture Comparison | Muneer et al. (2023) and Chinthulla (2025) demonstrate CNN+RNN outperforming 3D CNN on small shoplifting benchmarks (F1 ~0.90 vs 0.84). | Adopt 2D CNN (ResNet18) + GRU. |
| 2 | Recurrent Cell Choice | Empirical tests on small surveillance datasets show GRU achieves equal or higher test accuracy than Transformer/LSTM with fewer parameters. | Use GRU; avoid heavier Transformers. |
| 3 | Augmentation Consistency | Per-frame independent augmentations destroy motion semantics. `torchvision.transforms.v2` enforces identical transforms across all $T$ frames. | Pass full $(T, C, H, W)$ tensor directly to `v2` transforms. |
| 4 | Data Leakage Defense | Random frame/clip splitting yields artificially inflated metrics (e.g. 98%) due to correlated background cues. | Strictly split by source video ID. |
| 5 | Class Imbalance | Multi-method stacking (oversampling + high loss weights) induces severe precision collapse (false alarms). | Use single clean mathematical solution: `pos_weight = 1.64` in `BCEWithLogitsLoss` + Validation Threshold Tuning. |
| 6 | Outlier Clip Mitigation | One normal shopping clip reaches 74.0s (vs 12-14s median). | Handled natively by fixed $T=16$ uniform sampling; optional multi-clip sliding window at inference for portfolio excellence. |
| 7 | Advanced Research Scalability | SlowFast and Pose-based methods (Shopformer) are powerful but break task tensor contracts or increase dependencies. | Retain as documented future work in README/conclusion to showcase senior-level research literacy. |
| 8 | Dataset Source Leakage Audit | Diagnostic B revealed 100% of validation clips had cosine similarity 1.0000 to train clips due to parent video sub-clipping (`_1.mp4`). | Enforce Group-Aware Split (`Cell 3.1`) grouping near-duplicates and parent clips into single splits. |

### 4.3 Master Implementation Blueprint (Cell-by-Cell Execution Contract)

To maintain absolute continuity for any Agent, the project execution follows this exact cell-by-cell sequence in the main Kaggle notebook (`shoplifting-detection-pipeline`):

* `Cell 0.0 & 1.0` — **Executive Overview, Architecture & Hardware Telemetry** [COMPLETED]
  * Deterministic seeds (`seed=42`), 2x Tesla T4 GPU detection (29 GB VRAM), dynamic `/kaggle/input/` dataset path detection.
* `Cell 2.0` — **Video Dataset Discovery & Exploratory Data Analysis (EDA)** [COMPLETED]
  * Verified audit: 855 total clips (531 Normal / 324 Theft), 100% uniform resolution (`704x576`), median duration 12-14s (max 74s).
* `Cell 2.1` — **Visual Inspection of Sample Clips & Frame Grids** [CURRENT STEP]
  * Render interactive video player for normal vs theft clips; display side-by-side $2 \times 8$ frame grids of extracted $T=16$ sequences.
* `Cell 3.0` — **Leakage-Free Stratified Train / Val / Test Split**
  * 70% Train (598 clips), 15% Val (128 clips), 15% Test (129 clips) grouped by video source/camera ID to prevent spatial background leakage.
* `Cell 4.0` — **Fixed Uniform Frame Sampling & Offline Disk Caching Engine**
  * Sample $T=16$ uniform frames per clip ($\text{indices} = \text{round}(\text{linspace}(0, N-1, 16))$); decode ONCE and cache to `/kaggle/working/cached_frames/`.
* `Cell 5.0` — **Clip-Consistent Spatial Augmentation & PyTorch Video Dataset**
  * `torchvision.transforms.v2` pipeline applying identical flip, crop, and jitter across all 16 frames in one call. Tensor contract: `(16, 3, 224, 224)`.
* `Cell 6.0` — **Hybrid Spatiotemporal Architecture (ResNet18 + GRU)**
  * Pretrained ResNet-18 (512-d spatial features) + 2-layer Bidirectional GRU (hidden dim: 256, dropout: 0.3) + Linear classification head.
* `Cell 7.0` — **Two-Stage Training & Optimization Loop**
  * Stage 1: Frozen backbone, train GRU head (`AdamW`, `lr = 1e-3`, `pos_weight = 1.64`).
  * Stage 2: Unfreeze `layer4` with lower `lr` (`5e-5`), keeping backbone batch norm in eval mode. Best model checkpoint saved to disk.
* `Cell 8.0` — **Validation Threshold Tuning & Rigorous Test Benchmark**
  * Optimize threshold $p^* \in [0.1, 0.9]$ on validation F1 curve. Report final test Accuracy, Precision, Recall, F1, PR-AUC, and Confusion Matrix.
* `Cell 9.0` — **Advanced Inference Upgrade: Sliding Window Multi-Clip Averaging**
  * Evaluate long outlier clips ($> 20$s) using overlapping 16-frame sliding windows with logit averaging, demonstrating senior-level production polish.

### 4.4 Completed Milestones & Live Status Tracker

| Step | Notebook Cell | Objective & Technical Focus | Key Metric / Result | Status |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `Cell 0.0 & 1.0` | Environment setup, deterministic seeds, GPU telemetry | 2x Tesla T4 (29 GB VRAM) verified | **Completed** |
| 2 | `Cell 2.0` | Container metadata audit, resolution & duration analysis | 531 Normal vs 324 Theft; 100% 704x576 | **Completed** |
| 3 | `Cell 2.1` | Visual inspection of sample clips & $2 	imes 8$ frame grids | Interactive player & frame grids | **Completed** |
| 4 | `Cell 3.0` | 70/15/15 grouped split by source video with stratification | 612 Train / 122 Val / 121 Test (Zero Leakage) | **Completed** |
| 5 | `Cell 4.0` | Offline caching of $T=16$ uniform frames per video | 855 cached clips @ 224x224 (1.92 GB in 2m 8s) | **Completed** |
| 6 | `Cell 5.0` | `torchvision.transforms.v2` consistent video dataset | Output tensor `(32, 16, 3, 224, 224)` | **Completed** |
| 7 | `Cell 6.0` | ResNet18 + GRU temporal model definition | Forward pass verified: shape torch.Size([2]), 13.6M params | **Completed** |
| 8 | `Cell 7.0` | Two-stage training (frozen head -> layer4 fine-tuning) | Best Val F1: 0.8696 (Val Acc: 90.16%), saved `best_model.pt` | **Completed** |
| 9 | `Cell 8.0` | Validation threshold search & test evaluation | Test Acc: 92.56%, Macro F1: 0.9053, ROC-AUC: 0.9755 | **Completed** |
| 10 | `Cell 9.0` | Production ONNX Export & Dynamic INT8 Quantization | FP32: 51.91 MB, INT8: 19.81 MB (61.8% compression, Parity < 1e-6) | **Completed** |
| 11 | `Deliverable` | GitHub clean repo, README with figures and results | Final portfolio delivery | **Next Step** |
### 4.5 Engineering Rationale & Architectural Q&A Ledger
Any Agent must consult these established design choices before proposing changes:
1. **Why Fixed Uniform $T = 16$ Frames?**
   Retail clips center around 12-14 seconds. Uniform sampling extracts an optical snapshot every ~0.75s, capturing the complete behavioral arc (enter $\rightarrow$ reach $\rightarrow$ conceal $\rightarrow$ depart) while maintaining a strict batch-friendly tensor geometry `(16, 3, 224, 224)`.
2. **How is the 74-Second Duration Outlier Handled?**
   Normalized automatically by the uniform sampling index math: $\text{round}(\text{linspace}(0, N-1, 16))$. In Section 9.0, an advanced sliding window test inference will optionally evaluate multi-clip segments to benchmark against uniform sampling.
3. **Why Single Loss Weighting (`pos_weight = 1.64`) instead of Oversampling?**
   Stacking duplicate oversampling on top of loss penalty creates a double-penalty distortion, causing severe false positive inflation (collapsing Precision). Single mathematical weighting in `BCEWithLogitsLoss` + Validation Threshold Tuning provides the clean, uncorrupted solution.
4. **Why Offline Frame Caching to `/kaggle/working/cached_frames/`?**
   OpenCV decoding of 855 compressed `.mp4` containers on every epoch creates an extreme disk-IO bottleneck. Decoding once and saving pre-resized arrays reduces epoch training time from ~4 minutes to ~12 seconds.
5. **Why Not SlowFast or Pose Estimation (Shopformer) in the Core Pipeline?**
   The task PDF strictly mandates a single `(T, C, H, W)` tensor and either a 3D CNN or 2D CNN + RNN. SlowFast requires dual-stream inputs $(T_{\text{slow}}, T_{\text{fast}})$, and Pose requires heavy external dependencies (MediaPipe/YOLO-Pose). Both are documented in the final report as future research to showcase senior literacy without violating task contracts.

### 4.4 Approved Restore Points & Checkpoints

| Tag / Commit | Identifier | Description & Repository State | Rollback Command |
| :--- | :---: | :--- | :--- |
| `v1.0-init` | `HEAD` | Scaffolding, updated AGENTS.md with fixed T=16 caching & Kaggle setup | `git reset --hard v1.0-init` |

### 4.5 Known Traps & Lessons Learned (To Prevent Future Mistakes)
1. **Video Decoding Bottleneck:** Never seek with `cv2.CAP_PROP_POS_FRAMES` or loop decode frame-by-frame on the fly during training. Decode once and cache the $T=16$ uniform frames to disk.
2. **Batch Size vs VRAM:** 3D CNNs cause OOM easily. ResNet18+GRU allows clean batch sizes of 16-32.
3. **Clip Augmentation Parity:** Never use independent frame transforms. Always pass `(T, C, H, W)` to `torchvision.transforms.v2`.
4. **Leakage Trap:** Never randomly shuffle clips before splitting if multiple clips share a parent recording. Split by source ID.
5. **Threshold Default Trap:** Never assume $0.5$ threshold is optimal for imbalanced anomaly detection. Always tune threshold on validation F1.

---

## 5. Summary Checklist for Every Task Execution
Before proposing completion of any step, the Agent must verify:
- [ ] Was the implementation explicitly authorized by the user?
- [ ] Is all code, markdown, and documentation 100% English?
- [ ] Does every code cell have a preceding, sequentially numbered Markdown documentation cell?
- [ ] Is the code concise, readable, and free of AI bloat?
- [ ] Are all reported metrics verified from real execution?
- [ ] Is `git push` withheld until explicit user request?
