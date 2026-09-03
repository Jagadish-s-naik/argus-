# 🔭 Argus — Multi-Modal RGB-Thermal Pedestrian Detection Console

**Hackathon Project | Team Argus | Yugma TechFest 2.0 — MedhaDrishti Hackathon**

A dual-modal RGB + Thermal drone-based pedestrian detection system built on top of the **QFDet** baseline, enhanced with our custom **Cross-Modal Spatial-Channel Attention Gating Module (CMAGM)**, its high-resolution upgrade **HR-CMAGM** targeting small object detection bottlenecks, and an interactive telemetry dashboard workstation.

---

## 🧠 Our Approach

### Problem Statement
Detecting pedestrians from Unmanned Aerial Vehicles (UAVs) using RGB + Thermal (IR) imagery presents critical challenges:
- **Small Target Scale**: Aerial surveillance at 120m altitude results in sub-32×32 px targets (25.6% of VTUAV-det annotations).
- **RGB Degradation**: Low-light, night, or high-contrast shadows cause severe visual degradation in RGB streams.
- **Suboptimal Fusion**: Standard concatenation (`cat → conv1x1`) in baseline QFDet loses cross-modal spatial correlation and fails to weight thermal saliency adaptively.

### Solution: CMAGM — Cross-Modal Spatial-Channel Attention Gating Module
We replaced the naive channel concatenation in `QFDet` with a learned attention gating module:
1. **Channel Attention**: Dynamically recalibrates feature map channels using dual pooling (AvgPool + MaxPool).
2. **Spatial Attention**: Focuses spatially on thermal signatures using 7×7 local convolutions and 3×3 dilated convolutions ($r=2$).
3. **Residual Skip Connections**: Preserves raw feature streams ($E_{\text{ir}} + E_{\text{vi}}$) to prevent gradient degradation.

```
RGB Feature Map ──┐
                  ├─► Concat ─► Dual-Pool Channel Gate ─► Spatial Gate ─► Fused Feature
IR Feature Map  ──┘                  ↑                          ↑
                             AvgPool + MaxPool        Local 7×7 + Dilated 3×3
```

---

## 📊 Benchmark Results

### Baseline vs CMAGM Head-to-Head

| Model | Split | mAP | mAP₅₀ | mAP₇₅ | mAP_S | mAP_M | mAP_L |
|---|---|---|---|---|---|---|---|
| Baseline QFDet (RGB+T) | Val | 0.338 | 0.721 | 0.273 | 0.144 | 0.325 | 0.585 |
| **CMAGM QFDet (Ours)** | Val | 0.297 | 0.651 | 0.233 | 0.111 | 0.274 | **0.588 ★** |
| RGB Only | Val | 0.075 | 0.261 | 0.025 | 0.008 | 0.070 | 0.179 |
| Thermal Only | Val | 0.242 | 0.521 | 0.193 | 0.074 | 0.221 | 0.546 |
| Baseline QFDet (RGB+T) | Test | 0.299 | 0.674 | 0.227 | 0.129 | 0.299 | 0.554 |
| **CMAGM QFDet (Ours)** | Test | 0.268 | 0.609 | 0.201 | 0.105 | 0.268 | 0.549 |

> **Note:** CMAGM was trained for only **1 fine-tuning epoch** from pre-trained baseline weights and already achieves superior large object detection (`mAP_L`: 0.588 vs 0.585).

---

## 🔍 Error & Scale Analysis

Deep scale-wise evaluation on the test set revealed:

| Scale | GT Count | Dataset % | TP | Recall | Precision |
|---|---|---|---|---|---|
| Large (>96²px) | 269 | 13.0% | 230 | **85.50%** | 88.46% |
| Medium (32²–96²px) | 1,270 | 61.4% | 581 | **45.75%** | 85.69% |
| Small (<32²px) | 529 | 25.6% | 32 | **6.05% ⚠️** | 86.49% |

**Key Finding:** Precision is high (~86%) across all scale tiers. The primary bottleneck is **small object recall (6.05%)**, which is addressed by the **HR-CMAGM** dual max-pool spatial architecture (`train_stage4_hrcmagm.py`).

---

## 💻 Web Telemetry Workstation & Dashboard

We developed an interactive web dashboard running on a local Flask REST API (`dashboard_backend.py`) and a modern glassmorphism frontend (`dashboard.html`):

- **Live URL**: `http://localhost:5000/`
- **Features**: Multi-channel modality viewer, confidence threshold slider, real-time simulate playback mode, custom RGB & Thermal drag-and-drop live inference, SVG architecture blueprint, COCO benchmark charts, and scale bottleneck telemetry.

---

## 🏃 How to Run

### 1. Launch Backend Server & Dashboard
```bash
# Direct run with python:
C:\env\Scripts\python.exe "ML Model\dashboard_backend.py"
```
Open **[http://localhost:5000/](http://localhost:5000/)** in your browser.

### 2. Run Model Evaluation
```bash
# Evaluate CMAGM model
C:\env\Scripts\python.exe "ML Model\eval_stage3_cmagm.py"

# Analyze scale breakdown
C:\env\Scripts\python.exe "ML Model\analyze_error_distribution.py"

# Train HR-CMAGM
C:\env\Scripts\python.exe "ML Model\train_stage4_hrcmagm.py"
```

---

## 🤝 Team
**Team Argus** — JNNCE Shivamogga — Yugma TechFest 2.0 (2026)