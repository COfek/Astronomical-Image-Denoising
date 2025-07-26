# 🌌 Astronomical Image Denoising

This repository contains our final project for the course **Model-based Deep Learning (361-2-2320)** at Ben-Gurion University. The goal is to denoise astronomical images using modern optimization frameworks that integrate traditional priors and deep learning models.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Implemented Methods](#implemented-methods)
- [Results](#results)
- [Ablation Studies](#ablation-studies)
- [Honorable Mentions](#honorable-mentions)
- [References](#references)

---

## 🌠 Overview

Astronomical images often suffer from severe noise due to long exposure times, low photon counts, and sensor limitations. We investigate and compare three major model-based optimization frameworks for image denoising:

1. **Regularization by Denoising (RED)**
2. **Alternating Direction Method of Multipliers (ADMM)**
3. **Score Matching Denoising (SMD)**

We utilize both classical and deep learning-based denoisers (e.g., BM3D, Tikhonov, UNet, ViT) to evaluate the reconstruction performance on real telescope data (SDSS).

---

## 🧱 Project Structure

```
├── Data/
│   ├── dataloader.py
│   ├── simulate_astronomy_dataset.py
├── Models/
│   ├── unet.py
│   ├── vit.py
│   ├── dncnn.py
│   ├── BM3D.py
│   ├── TV.py
│   └── tikhonov.py
├── Scripts/
│   ├── train_val_test.py
│   ├── admm.py
│   ├── smd.py
│   ├── utils.py
│   ├── plots.py
├── notebooks/
│   └── inference_and_training.ipynb
├── output/
├── config.yaml
├── main.py
└── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/COfek/Astronomical-Image-Denoising.git
cd Astronomical-Image-Denoising
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🔹 Run All Pipelines from `main.py`:

```bash
python main.py
```

### 🔹 Alternatively, use the Jupyter Notebook:

You can run training and inference from the notebook provided in:

```
/notebooks/inference_and_training.ipynb
```

This is especially useful for debugging, visualization, and exploratory experiments.

---

## 🧠 Implemented Methods

### 🔸 Regularization by Denoising (RED)
- Solves: `min_x ½‖y - Hx‖² + λ/2 xᵀ(x - f(x))`
- Tested with: **UNet**, **ViT**, **DnCNN**, **BM3D**, **TV**, **Tikhonov**

### 🔸 ADMM
- Optimization-based inverse method with plug-and-play denoisers

### 🔸 Score Matching Denoising (SMD)
- Uses the score function of a denoiser (∇ log p(x)) to reconstruct clean images

---

## 📊 Results

We evaluate models using:
- **PSNR**
- **SSIM**
- Visual comparison

---

## 🔍 Ablation Studies

We analyze:
- The effect of denoiser choice
- RED vs ADMM vs SMD
- Impact of PSF blur and Gaussian noise

---

## 🧪 Honorable Mentions

Other models we tried:
- **RCAN**
- **RCAN-Swin**
- **Fusion Model**
- **SvOcSRCNN**
- **PyramidDeepSRCNN_CA**

---

## 📖 References

1. Romano et al. (2017). RED  
2. Zhang et al. (2017). Residual Learning for Denoising  
3. Kadkhodaie & Simoncelli (2021). Implicit Prior via Denoiser  
4. Adler & Öktem (2018). Learned Primal-Dual  

---

## 🧑‍💻 Authors

- **Ofek Cohen** (206713711)
- **Shaked Vaknin** (207472697)
- **Adi Doplet**

For more details, visit our [GitHub repo](https://github.com/COfek/Astronomical-Image-Denoising).
