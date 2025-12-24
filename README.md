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

- **Accuracy (mAP@0.5): 0.857**
- **mAP@0.5:0.95: 0.692**
- Results exceed or match common benchmark performance.

---

## 3.2 Training & Validation Loss Curves

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/loss_curves.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/loss_curves.png" width="900">
  </a><br>
  <em>Training and validation loss curves for Box, Classification, and DFL losses</em>
</p>

- Consistent loss reduction across epochs.
- Training and validation curves remain close.
- Indicates stable convergence with minimal overfitting.

---

## 3.3 Separated Loss Analysis

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/Losses.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/Losses.png" width="900">
  </a><br>
  <em>Separated training and validation loss trends</em>
</p>

- **Box Loss:** Smooth and steady decline.
- **Classification Loss:** Rapid early improvement.
- **DFL Loss:** Gradual reduction, improving localization quality.

---

## 3.4 Accuracy & Metrics Curves

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/metrics_accuracy_curves.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/metrics_accuracy_curves.png" width="950">
  </a><br>
  <em>Precision, Recall, mAP@0.5 (Accuracy), and mAP@0.5:0.95 vs Epoch</em>
</p>

- **Best mAP@0.5:** `0.857` at **Epoch 15**
- **Best mAP@0.5:0.95:** `0.692` at **Epoch 25**
- Precision and recall remain balanced throughout training.

---

## 3.5 Per-Class Performance

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/Per-class%20mAP.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/inference/viz_out/Per-class%20mAP.png">
  </a><br>
  <em>Per-class mAP@0.5:0.95 distribution</em>
</p>

<h2 align="center"><strong>4. Running the Demo App</strong></h2>

The project provides a simple demo application for running object detection using the fine-tuned **YOLOv11** model.  
Detection can be performed using **two different input sources**, depending on your use case.

---


## 4.1 Example: Detection via Video File

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/video.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/video.png" width="900">
  </a><br>
  <em>Sample video detection output — bounding boxes and class labels appear on every frame</em>
</p>

The model processes the video sequentially and consistently detects objects across the entire clip.

👉 **Click the image above to view results of detection via a video file**

---

## 4.3 Example: Detection via Webcam

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/live-cam.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/live-cam.png" width="900">
  </a><br>
  <em>Sample webcam detection output — real-time detection using a live camera feed</em>
</p>

In webcam mode, the model performs inference in real time, delivering fast and stable detection results.

---

## 4.4 CLI (Command Line Interface)

The demo application is executed via a **Command Line Interface (CLI)**, enabling easy control and integration.

### ▶️ Webcam Detection

```bash
python webcam-script.py --source webcam
```

This command:
- Starts real-time detection from the webcam
- Displays detection results live on screen

---

### ▶️ Video File Detection

```bash
python webcam-script.py --source video --path path/to/video.mp4
```

This command:
- Loads and processes the specified video file
- Displays bounding boxes and class labels on each frame

---

## 4.5 Terminal Output During Webcam Detection

<p align="center">
  <a href="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/CLI.png">
    <img src="https://github.com/MariHovhannisyan/YOLOv11-PascalVOC/blob/main/CLI.png" width="900">
  </a><br>
  <em>Terminal output showing inference speed, frame details, and detected objects</em>
</p>

During webcam detection, the terminal displays:
- Inference speed (FPS)
- Frame index and resolution
- Detected object classes with confidence scores

---

## 4.6 Summary

- The demo app supports both **real-time** and **offline** detection modes
- CLI-based execution ensures simplicity and flexibility
- The YOLOv11 model delivers **fast, stable, and reliable inference**
- Suitable for demos, testing, and real-world deployment scenarios

