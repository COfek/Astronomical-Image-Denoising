# 🔭 Astronomy Image Denoising with RED and Deep Learning

This project explores **image denoising and deblurring** techniques for astronomical images using both **classical** and **learned denoisers**, with and without **Regularization by Denoising (RED)**. It supports models like UNet, DnCNN, and ViT, and includes a modular pipeline for training, evaluation, and restoration via RED.

---

## 📁 Project Structure

```
├── Data/
│   ├── clean/                     # Clean astronomical images
│   ├── noisy/                     # Corresponding noisy images
│   ├── psf.npy                    # Point Spread Function (PSF) kernel
│   └── simulate_astronomy_dataset.py
├── Models/
│   ├── unet.py                    # UNet architecture
│   ├── DnCNN.py                   # DnCNN architecture
│   ├── ViT.py                     # Vision Transformer denoiser
│   ├── BM3D.py                    # Wrapper for classical BM3D
│   ├── TV.py                      # Total Variation denoiser
│   └── tikhonov.py                # Linear Tikhonov denoiser
├── Scripts/
│   ├── train_unet.py              # Training loop for learned models
│   ├── red_inference.py           # RED iterative restoration
│   ├── plots.py                   # Plotting and visualization functions
│   └── utils.py                   # PSNR, SSIM and general utilities
├── output/
│   └── ...                        # Saved models, plots, metrics
├── main.py                        # Main script for training/evaluation
```

---

## 🚀 Features

- ✅ Support for classical denoisers: **BM3D**, **TV**, **Tikhonov**
- 🧠 Deep denoisers: **UNet**, **DnCNN**, **ViT**
- 🔁 Integration with **RED (Regularization by Denoising)**
- 📊 Metrics: **PSNR** and **SSIM**
- 🖼️ Side-by-side visualizations and quantitative comparisons
- 📦 JSON export of results for analysis/reporting

---

## 🧪 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision matplotlib tqdm opencv-python
```

(Optional: install `bm3d` if using BM3D wrapper)

---

### 2. Simulate or Prepare Dataset

If you don't already have the dataset:

```bash
python main.py --download
```

Otherwise, ensure your data is in `Data/clean` and `Data/noisy`.

---

### 3. Train a Model

To train a model (e.g., UNet):

```python
# Inside main.py
DO_TRAIN = True
MODEL_TO_TRAIN = "UNet"  # or "DnCNN", "ViT"
```

Then run:

```bash
python main.py
```

---

### 4. Evaluate & Run RED

To run inference and RED restoration (no training):

```python
DO_TRAIN = False
MODEL_TO_TRAIN = "DnCNN"
```

---

## 🖼️ Outputs

- `output/<model>/best_<model>.pth` – trained model
- `output/red_results/` – comparison plots and metrics JSON
- `output/<model>/*.png` – training/validation loss, PSNR, SSIM curves

---

## 🧠 Models

### UNet
Fully convolutional encoder-decoder with skip connections.

### DnCNN
Residual learning with batch normalization for denoising.

### ViT
Vision Transformer adapted for image-to-image denoising.

---

## 📚 RED: Regularization by Denoising

RED solves inverse problems by using a denoiser as a prior:
> \( x^* = rg \min_x \ell(x; y) + \lambda ho_{	ext{RED}}(x) \)

where \( ho_{	ext{RED}}(x) = rac{1}{2} x^	op (x - f(x)) \), and \( f(x) \) is a denoising operator.

Implemented via:
- RED Gradient Descent
- RED-ADMM (configurable)
- RED with learned and classical denoisers

---

## 📈 Example Results

| Method             | PSNR (dB) | SSIM  |
|--------------------|-----------|--------|
| Direct UNet        | 32.42     | 0.912 |
| RED-BM3D           | 28.53     | 0.831 |
| RED-UNet           | 29.87     | 0.854 |
| RED-Tikhonov       | 27.66     | 0.792 |

> Results saved in `output/red_results/*.json` and plotted side-by-side

---

## 🧩 TODOs / Ideas

- [ ] Add support for other plug-and-play priors (e.g., DRUNet)
- [ ] Implement RED-PRS / RED-GEC variants
- [ ] Extend to blind PSF estimation
- [ ] Benchmark on real astronomical datasets

---

## 📝 Citation

If you use this code, please cite relevant works:
- Romano et al., *RED: Regularization by Denoising*, 2017
- Zhang et al., *DnCNN: Beyond a Gaussian Denoiser*, 2017
- Ronneberger et al., *UNet*, 2015

---

## 👤 Author

Developed by [Your Name]  
M.Sc. Student @ Ben-Gurion University, IDF Engineer  
Contact: [your.email@example.com]
