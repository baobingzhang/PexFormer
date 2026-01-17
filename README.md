# PexFormer: Robust Indoor Human Localization via Patch-level Tokenization and Semi-Permeable Attention

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> **Official Implementation for ICSR 2026 Submission**

## 📖 Abstract

**Non-intrusive indoor human localization** plays a pivotal role in the development of next-generation Ambient Assisted Living (AAL) systems. However, existing data-driven methods often struggle with extreme class imbalance—characterized by long-tail distributions of room occupancy—and the inherent noise of sparse sensor triggers.

To address these challenges, we propose **PexFormer**, a novel architecture that adapts efficient computational paradigms from the tabular domain to time-series localization tasks. PexFormer leverages **Patch-level Tokenization** to effectively capture local temporal dynamics and integrates a **Semi-Permeable Attention (SPA)** mechanism to construct hierarchical feature interactions.

**Key Innovation**: We empirically demonstrate that a simple **Random Permutation** strategy within the SPA framework significantly outperforms traditional Mutual Information-based sorting, effectively serving as a robust regularization technique against overfitting.

## ✨ Key Features

- **Patch-Level Tokenization**: Treats sensor time windows as atomic patches, reducing sequence length and computational complexity quadratic in terms of sequence length $L$ to quadratic in terms of the number of patches $L/P$.
- **Semi-Permeable Attention (SPA)**: A structured attention mechanism that filters noise and enforces hierarchical feature interaction.
- **Random Permutation Regularization**: A counter-intuitive discovery that random feature ordering outperforms mutual information sorting for time-series patches.
- **SOTA on Imbalanced Data**: Achieves exceptional Macro-F1 scores without complex resampling.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Preparation

Data obtained as required and Place your sensor data (Excel format) in the `data/` directory.

### Usage

**1. Training PexFormer**
```bash
python run_Pexformer.py 
```

**2. Ablation Study**
```bash
python run_Patch_ExcelFormer_ablation.py
```

**3. Visualization**
```bash
python visualization.py
```


## 📊 Key Results

| Model | Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| BiLSTM | 85.7% | 46.46% |
| NODE | 83.6% | 33.3% |
| TabTransformer | 80.8% | 33.6% |
| DCN V2 | 83.4% | 39.5% |
| PexFormer (Ours) | **93.3%** | **83.52%** |


