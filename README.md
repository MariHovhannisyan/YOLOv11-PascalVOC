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
  <a href="[https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/benchmark.png]">
    <img src="[benchmark-img]" width="850">
  </a><br>
  <em>Comparison of best achieved results against typical benchmark ranges</em>
</p>

**Summary:**
- **Accuracy (mAP@0.5): 0.857**
- **mAP@0.5:0.95: 0.692**
- Performance exceeds or matches the upper bound of common benchmark ranges.

---

## 3.2 Training & Validation Loss Curves

<p align="center">
  <a href="[loss-curves-img]">
    <img src="[loss-curves-img]" width="900">
  </a><br>
  <em>Training and validation loss curves for Box, Classification, and DFL losses</em>
</p>

**Observations:**
- Steady decrease in both training and validation losses.
- No visible divergence between train and validation curves.
- Indicates stable convergence with minimal overfitting.

---

## 3.3 Separated Loss Analysis

<p align="center">
  <a href="[separated-losses-img]">
    <img src="[separated-losses-img]" width="900">
  </a><br>
  <em>Separated training and validation loss trends</em>
</p>

**Details:**
- **Box Loss:** Gradual and consistent reduction.
- **Classification Loss:** Rapid early improvement, followed by refinement.
- **DFL Loss:** Smooth decline, improving bounding box quality.

---

## 3.4 Accuracy & Metrics Curves

<p align="center">
  <a href="[metrics-img]">
    <img src="[metrics-img]" width="950">
  </a><br>
  <em>Precision, Recall, mAP@0.5 (Accuracy), and mAP@0.5:0.95 vs Epoch</em>
</p>

**Key Results:**
- **Best Accuracy (mAP@0.5):** `0.857` at **Epoch 15**
- **Best mAP@0.5:0.95:** `0.692` at **Epoch 25**
- Balanced precision and recall across epochs.

---

## 3.5 Per-Class Performance

<p align="center">
  <a href="[per-class-img]">
    <img src="[per-class-img]" width="950">
  </a><br>
  <em>Per-class mAP@0.5:0.95 distribution</em>
</p>

**Insights:**
- Most classes achieve **mAP > 0.6**
- Multiple classes exceed **0.8 mAP**
- Lower-performing classes indicate potential data imbalance or complexity.

---



