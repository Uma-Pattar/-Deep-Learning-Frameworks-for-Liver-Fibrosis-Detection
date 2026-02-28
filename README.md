# 🧠 Deep Learning Frameworks for Liver Fibrosis Detection  
### Comparative CNN & Vision Transformer Study using Ultrasound Images(F0–F4)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)
![Status](https://img.shields.io/badge/Project-Research%20Grade-success)
![Domain](https://img.shields.io/badge/Domain-Medical%20AI-purple)

---

## 📌 Overview

This repository contains the complete implementation of a **research-grade comparative study** evaluating deep learning frameworks for multi-class liver fibrosis staging (F0–F4) using B-mode ultrasound images.

The project systematically compares CNN and Vision Transformer architectures under a unified pipeline to analyze:

- Predictive performance  
- Generalization ability  
- Computational efficiency  
- Clinical interpretability  

This work is based on the research paper:

> **A Comprehensive Comparative Study of Deep Learning Frameworks for Liver Fibrosis Detection Using B-mode Ultrasound Images**

---

## 🎯 Problem Statement

Liver fibrosis progresses across five stages:

| Stage | Description |
|------|-------------|
| F0 | No fibrosis |
| F1 | Mild fibrosis |
| F2 | Moderate fibrosis |
| F3 | Advanced fibrosis |
| F4 | Cirrhosis |

Ultrasound-based staging is difficult due to subtle texture variations between adjacent stages.  
This project explores whether deep learning can reliably learn these patterns.

---

## 🏗 Implemented Deep Learning Models

### 🔹 Custom CNN (Baseline)
- 4 convolutional blocks  
- BatchNorm + ReLU  
- Dropout regularization  
- Trained from scratch  

### 🔹 ResNet50 (Transfer Learning)
- ImageNet pretrained  
- Residual connections  
- Strong medical imaging baseline  

### 🔹 MobileNetV3-Large (Efficient Model)
- Lightweight architecture  
- Deployment-friendly  
- Excellent performance vs size trade-off  

### 🔹 EfficientNet-B0 (Best Performer)
- Compound scaling  
- 5-fold stratified cross-validation  
- Highest overall accuracy & macro-F1  

### 🔹 Dual-Branch Vision Transformer (Novel Contribution)
- RGB branch + CLAHE-enhanced branch  
- ViT-Tiny backbone  
- Feature fusion using MLP  
- Global context modeling  

---

## 🧪 Experimental Pipeline

### 📂 Dataset
- Liver fibrosis ultrasound dataset (Kaggle)
- 6,323 images
- 5-class classification (F0–F4)
- Class imbalance handled explicitly

### ⚙ Preprocessing
- Resize: 224×224  
- Normalization  
- Data augmentation:
  - Rotation
  - Brightness/contrast
  - Gaussian blur
  - Elastic deformation

### 🧠 CLAHE Enhancement
Contrast Limited Adaptive Histogram Equalization used for:
- Texture enhancement  
- Dual-branch transformer fusion input  

### ⚖ Class Imbalance Handling
- WeightedRandomSampler  
- Class-weighted cross-entropy  

### 🧮 Training Setup
- Optimizer: AdamW  
- Mixed Precision Training (AMP)  
- LR Scheduler: ReduceLROnPlateau  
- GPU training enabled  

---

## 📊 Evaluation Metrics

- Accuracy  
- **Macro-F1 (primary metric)**  
- AUC (one-vs-rest multiclass)  
- Confusion matrix  
- Precision/Recall per class  

Macro-F1 used due to class imbalance and clinical relevance.

---

## 📈 Results Summary

| Model | Accuracy | Macro-F1 | AUC |
|------|---------|---------|------|
| EfficientNet-B0 | **0.9788** | **0.9711** | **0.9984** |
| ResNet50 | 0.9731 | 0.9645 | 0.9968 |
| MobileNetV3 | 0.9747 | 0.9630 | 0.9983 |
| Dual-Branch ViT | 0.9486 | 0.9293 | 0.9920 |
| Custom CNN | 0.6466 | 0.5755 | 0.8970 |

### 💡 Key Insights
- Transfer learning >> training from scratch  
- EfficientNet shows best generalization  
- MobileNetV3 best lightweight deployment model  
- ViT fusion promising but data-sensitive  
- F2 stage most difficult due to clinical overlap  

---

## 🔎 Explainable AI (Grad-CAM)

Grad-CAM applied on CNN models to:
- Visualize model attention regions  
- Ensure focus on liver parenchyma  
- Improve interpretability for clinical AI  

---

## 💻 Tech Stack

- Python  
- PyTorch  
- timm  
- Albumentations  
- OpenCV  
- NumPy & Pandas  
- Matplotlib  

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Deep-Learning-Frameworks-for-Liver-Fibrosis-Detection.git
cd Deep-Learning-Frameworks-for-Liver-Fibrosis-Detection
pip install -r requirements.txt
```

---

## 🚀 Training

```bash
python train_resnet50.py
python train_mobilenet.py
python train_efficientnet.py
python train_dual_vit.py
```

---

## 📁 Repository Structure

```
custom_cnn/
resnet50/
mobilenetv3/
efficientnet/
dual_branch_vit/
gradcam/
utils/
results/
```

---

## 🚀 Future Improvements

- Larger Vision Transformers  
- External dataset validation  
- Clinical deployment optimization  
- Multi-modal ultrasound fusion  
- Semi-supervised learning  

---

## 📜 Citation

```
@article{liverfibrosis_dl_2026,
title={Comparative Deep Learning Frameworks for Liver Fibrosis Detection},
author={Uma MP et al.},
year={2026}
}
```

---

## 👩‍💻 Author

**Uma MP**  
AI & Data Science Engineer  
Deep Learning • Medical Imaging • NLP
