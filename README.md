<h1 align="center"><strong>YOLOv11 Object Detection (Pascal VOC): Training Logs, Metrics & Inference</strong></h1>

---

<h2 align="center"><strong>1. Project Overview</strong></h2>

This project contains a fine-tuned <strong>YOLOv11</strong> object detector for the <strong>PASCAL VOC</strong> dataset (20 classes), along with:
- trained model weights in multiple formats (<code>.pt</code>, <code>.onnx</code>, <code>.engine</code>)
- training log (<code>.csv</code>)
- scripts for inference (webcam) and a notebook for experiments
- (optional) generated plots, stored separately (not part of the core project structure)

---

<h2 align="center"><strong>2. Project Structure</strong></h2>

<pre>
YOLOv11-Obj-Det-PascalVOC/
├─ inference/
│  └─ models/
│     ├─ yolo11m-finetune.pt
│     ├─ yolo11m-finetune.onnx
│     └─ yolo11m-finetune.engine
├─ yolo11m-finetune.csv
├─ metrics.py
├─ webcam-script.py
├─ YOLOv11-obj-det-PascalVOC.ipynb
├─ CLI.png
├─ live-cam.png
└─ video.png
</pre>

<h2 align="center"><strong>3. Results & Visualizations</strong></h2>


## 3.1 Benchmark Comparison

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/benchmark.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/benchmark.png" width="850">
  </a><br>
  <em>Comparison of best achieved results against typical benchmark ranges</em>
</p>

**Summary:**
- **Accuracy (mAP@0.5): 0.857**
- **mAP@0.5:0.95: 0.692**
- Results exceed or match common benchmark performance.

---

## 3.2 Training & Validation Loss Curves

<p align="center">
  <a href="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/loss_curves.png">
    <img src="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/loss_curves.png" width="900">
  </a><br>
  <em>Training and validation loss curves for Box, Classification, and DFL losses</em>
</p>

**Observations:**
- Consistent loss reduction across epochs.
- Training and validation curves remain close.
- Indicates stable convergence with minimal overfitting.

---

## 3.3 Separated Loss Analysis

<p align="center">
  <a href="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/Losses.png">
    <img src="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/Losses.png" width="900">
  </a><br>
  <em>Separated training and validation loss trends</em>
</p>

**Details:**
- **Box Loss:** Smooth and steady decline.
- **Classification Loss:** Rapid early improvement.
- **DFL Loss:** Gradual reduction, improving localization quality.

---

## 3.4 Accuracy & Metrics Curves

<p align="center">
  <a href="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/metrics_accuracy_curves.png">
    <img src="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/metrics_accuracy_curves.png" width="950">
  </a><br>
  <em>Precision, Recall, mAP@0.5 (Accuracy), and mAP@0.5:0.95 vs Epoch</em>
</p>

**Key Results:**
- **Best mAP@0.5:** `0.857` at **Epoch 15**
- **Best mAP@0.5:0.95:** `0.692` at **Epoch 25**
- Precision and recall remain balanced throughout training.

---

## 3.5 Per-Class Performance

<p align="center">
  <a href="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/Per-class%20mAP.png">
    <img src="https://raw.githubusercontent.com/USERNAME/YOLOv11-Obj-Det-PascalVOC/main/viz_out/Per-class%20mAP.png" width="950">
  </a><br>
  <em>Per-class mAP@0.5:0.95 distribution</em>
</p>