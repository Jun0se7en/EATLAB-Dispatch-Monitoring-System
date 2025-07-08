# Dispatch Monitoring System (DMS)

Real-time visual inspection for the *dispatch* line of a dessert shop.  
DMS detects every tray / dish that passes the camera, classifies its
state (*kakigori*, *empty*, …), tracks the item across frames and keeps
running counters.  The Streamlit demo lets operators correct wrong
predictions; all feedback is stored for future fine-tuning.

---

## 1  Project Highlights
| Module | Approach | Notes |
|--------|----------|-------|
| **Detection** | Fine-tuned **YOLOv8-l** | mAP<sub>50</sub> 0.88 (5-fold CV) |
| **Classification** | Fine-tuned **ResNet-50** with **Focal Loss** | tackles heavy class imbalance, macro-F1 0.93 |
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
