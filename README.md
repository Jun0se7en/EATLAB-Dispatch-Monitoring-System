# Dispatch Monitoring System (DMS)

Real-time visual inspection for the *dispatch* line of a dessert shop.  
DMS detects every tray / dish that passes the camera, classifies its
state (*kakigori*, *empty*, *not_empty*), tracks the item across frames and keeps
running counters.  The Streamlit demo lets operators correct wrong
predictions; all feedback is stored for future fine-tuning.

---

## 1  Project Highlights
| Module | Approach | Notes |
|--------|----------|-------|
| **Detection** | Fine-tuned **YOLOv8-l** | mAP<sub>50</sub> 99.385% |
| **Classification** | Fine-tuned **ResNet-50** with **Focal Loss** | tackles heavy class imbalance, Accuracy on validation set 99.776% (Best of 5-fold) |
| **Tracking** | **Deep SORT Realtime** | GPU-accelerated, stable IDs |
| **Data boost** | Brightness / contrast / hue / saturation augments | ↑ diversity without new shoots |
| **Validation** | **Stratified K-Fold (5)** | choose best fold ➜ export models |
| **User feedback loop** | ROI chooser, per-object label correction | crops + metadata saved to `feedback/` |

---

## 2  Repository Layout

```text
.
├─ application.py          ← Streamlit Demo Application
├─ models/
│  ├─ detection_model.pt    (YOLOv8l)
│  ├─ classification_model.pt (RESNet)
│  └─ classification_model.onnx
├─ feedback/               ← new crops + JSONL go here
├─ demo/                   ← mini sample video for quick test
├─ utils/                  ← folder contains training code for classification task and detection task
├─ docker-compose.yml
├─ Dockerfile
└─ README.md
```

---

## 3 Quick Start

### 3.1 Prerequisites

Docker Engine

### 3.2 Build & Run

```bash
git clone https://github.com/Jun0se7en/EATLAB-Dispatch-Monitoring-System.git
cd EATLAB-Dispatch-Monitoring-System

docker compose up --build
```

---
When the log shows

```bash
You can now view your Streamlit app in your browser.
  URL: http://0.0.0.0:8501
```

---
open http://localhost:8501

---

## 4 Demo Workflow
### 1. Upload a video clip
### 2. Draw ROI on first frame -> press Confirm ROI
<p align="center"> <img src="/demo/Input&CropROI.gif" width="48%"/> </p>

### 3. Monitoring live detections, IDs and counters
<p align="center"> <img src="/demo/Feedback.gif" width="48%"/> </p>

### 4. Whenever a new ID appears or its label changes, the app pops up a feedback card: pick the correct label -> Save. The crop image (feedback/crops/…png) and a JSONL line are appended.
<p align="center"> <img src="/demo/Demo.gif" width="48%"/> </p>