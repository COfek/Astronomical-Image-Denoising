# 🌌 Astronomical Image Denoising — UNet Denoiser Branch

This branch of the repository implements a deep learning-based approach for denoising astronomical images using a U-Net architecture. It is designed to restore high-fidelity observations of deep-space targets, such as galaxies and nebulae, by removing synthetic or real noise from telescope images.

---

## 📁 Project Structure

Astronomical-Image-Denoising/
│
├── data/ # Contains training/validation/test sets
├── models/ # Saved UNet model checkpoints
├── outputs/ # Denoised image results and comparisons
├── scripts/
│ ├── train_unet.py # Training script for UNet
│ ├── evaluate_unet.py # Evaluation script (PSNR, SSIM)
│ └── utils.py # Helper functions (metrics, visualization)
├── configs/
│ └── unet_config.yaml # Configuration file for training params
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🧠 Model Overview

We use the [U-Net](https://arxiv.org/abs/1505.04597) architecture, originally developed for biomedical image segmentation, adapted for denoising:

- **Encoder**: Captures hierarchical features using convolutional blocks with downsampling.
- **Decoder**: Reconstructs the image using transposed convolutions and skip connections.
- **Skip Connections**: Preserve spatial details lost during downsampling.

---

## 🧪 Metrics

To evaluate denoising performance, we use:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- Qualitative visual comparison (before vs. after)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone -b unet-denoiser https://github.com/COfek/Astronomical-Image-Denoising.git
cd Astronomical-Image-Denoising

2. Create a Virtual Environment
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt
