# ViLT PathVQA — Fast & Optimized Pathology Visual Question Answering Pipeline

This repository contains a highly optimized, high-performance training and inference pipeline for **Pathology Visual Question Answering (PathVQA)** using the **Vision-and-Language Transformer (ViLT)** (`vilt-b32-mlm`). 

The pipeline is explicitly tailored for binary (Yes/No) clinical and diagnostic questions on pathology images, utilizing advanced PyTorch optimization techniques to maximize GPU efficiency and prevent aggressive overfitting.

## 🚀 Key Features & Optimizations

* **High-Resolution Medical Processing:** Standardized processing at ViLT's native $384 \times 384$ resolution to preserve critical microscopic, cellular, and tissue-level structures.
* **Pathology-Specific Data Augmentations:** Integrated orientation-agnostic transforms (`RandomHorizontalFlip`, `RandomVerticalFlip`, and `RandomRotation`) to mimic variations in biopsy slide preparation.
* **GPU-Optimized Performance:** Uses Automatic Mixed Precision (`torch.amp.autocast`), PyTorch `GradScaler`, page-locked memory (`pin_memory=True`), and persistent dataloader workers to eliminate I/O bottlenecks.
* **Robust Regularization:** Enhanced with custom dropout layers (increased to `0.2`) and strategic partial layer unfreezing (`UNFREEZE_LAST_N=4`) to stabilize learning on smaller medical datasets.
* **Accuracy-Driven Checkpointing:** Features an early-stopping mechanism that monitors validation accuracy to capture peak performance before overfitting occurs.

---

## 📈 Performance Summary

The modifications to the pipeline shifted the training dynamics from instant overfitting to a robust, smooth optimization curve, yielding strong generalization metrics:

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **~86.5%** |
| **Best Validation Loss** | **0.6604** |
| **Training Flow** | Smooth convergence (no instant memorization) |

### Per-Class Diagnostic Performance
```text
              precision    recall  f1-score   support

         yes       0.88      0.88      0.88      1816
          no       0.85      0.86      0.86      1546
