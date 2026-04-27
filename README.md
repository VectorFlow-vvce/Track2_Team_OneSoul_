# 🌍 Offroad Terrain Segmentation (DINOv2 + ConvNeXt Head)

## 🚀 Overview

This project focuses on **semantic segmentation of off-road environments** using a powerful combination of:

* **DINOv2 Vision Transformer (ViT)** as a feature extractor
* **Custom ConvNeXt-style segmentation head** for pixel-wise classification

The goal is to accurately segment terrain into multiple classes such as trees, bushes, rocks, sky, etc., achieving high IoU performance for real-world applications like autonomous navigation.

---

## 📊 Features

* ✅ Multi-class segmentation (11 classes)
* ✅ Boundary-aware loss for sharper predictions
* ✅ Dice Loss + CrossEntropy combination
* ✅ Hard Example Mining (focus on difficult samples)
* ✅ Copy-Paste augmentation for rare classes (flowers, logs)
* ✅ Dynamic class weighting (boost weak classes)
* ✅ Multi-resolution training
* ✅ Feature caching (CPU speed optimization)

---

## 🗂️ Project Structure

```
Offroad_Segmentation_Scripts/
│
├── train_segmentation.py        # Main high-performance training script
├── test_segmentation.py         # Evaluation & inference
├── visualize.py                 # Visualization of predictions
│
├── ENV_SETUP/
│   ├── create_env.bat
│   ├── install_packages.bat
│   └── setup_env.bat
│
├── predictions/
│   ├── comparisons/             # Input vs prediction visuals
│   ├── masks/                   # Raw predicted masks
│   └── masks_color/             # Colored segmentation maps
│
├── train_stats/
│   ├── training_curves.png
│   ├── iou_curves.png
│   ├── dice_curves.png
│   └── evaluation_metrics.txt
│
└── segmentation_head.pth        # Trained model weights
```

---

## 🧠 Model Architecture

* **Backbone:** DINOv2 ViT (frozen)
* **Head:** ConvNeXt-inspired CNN
* **Input:** RGB image
* **Output:** Pixel-wise class prediction (11 classes)

---

## 🏷️ Classes

| ID | Class          |
| -- | -------------- |
| 0  | Background     |
| 1  | Trees          |
| 2  | Lush Bushes    |
| 3  | Dry Grass      |
| 4  | Dry Bushes     |
| 5  | Ground Clutter |
| 6  | Flowers 🌸     |
| 7  | Logs 🪵        |
| 8  | Rocks          |
| 9  | Landscape      |
| 10 | Sky            |

---

## ⚙️ Setup Instructions

### 1. Create Environment

```
ENV_SETUP/create_env.bat
```

### 2. Install Dependencies

```
ENV_SETUP/install_packages.bat
```

### 3. Activate Environment

```
ENV_SETUP/setup_env.bat
```

---

## 🏋️ Training

### Standard Training

```
python train_segmentation.py
```

### Fast CPU Mode (Feature Caching)

* Extract features once
* Train only segmentation head

---

## 📈 Metrics

* **IoU (Intersection over Union)** — primary metric
* **Pixel Accuracy**
* **Dice Score**

---

## 🎯 Performance

* Baseline IoU: **0.44**
* Target IoU: **0.70+**
* Fast CPU Mode: **0.55 – 0.65 (within ~30–60 mins)**

---

## 🔍 Inference

```
python test_segmentation.py
```

Outputs:

* Predicted masks
* Colorized segmentation
* Comparison visualizations

---

## 🧪 Key Innovations

* Boundary-aware learning for edge precision
* Rare-class boosting using Copy-Paste
* Dynamic difficulty-based sampling
* Lightweight head for fast CPU training

---

## 📌 Notes

* Backbone is frozen → faster training
* GPU recommended but CPU mode supported
* Large datasets should NOT be committed to Git

---

## 🤝 Future Improvements

* Real-time inference optimization
* Deployment (ONNX / TensorRT)
* Integration with autonomous navigation systems

---

## 👩‍💻 Author

**Rachana N**

---

## ⭐ If you found this useful

Consider giving a ⭐ to the repository!
