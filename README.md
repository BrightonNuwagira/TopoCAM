# 🧠 TopoCAM: ROI-Driven Topological Signatures in Medical Imaging

TopoCAM is a modular framework that combines explainable deep learning with topological data analysis (TDA) for interpretable and robust medical image classification. It supports both **3D volumetric scans** and **2D grayscale images**, and has been validated across multiple MedMNIST and BraTS benchmarks.

---

## 🔍 Motivation

Deep learning models often struggle with interpretability and robustness in clinical settings, especially under limited supervision. TopoCAM addresses this by:

- Localizing class-discriminative regions using **multi-scale Grad-CAM**
- Segmenting the input image/volume based on fused attention maps
- Computing **Betti curves** via cubical persistent homology
- Classifying topological descriptors using a lightweight MLP

This approach filters out irrelevant anatomy and concentrates analysis on clinically meaningful structures.

---

## 🧩 Pipeline Overview

The TopoCAM pipeline is structured as:
