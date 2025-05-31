import os
import requests
from io import BytesIO
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import convolve2d
from tqdm import tqdm  # Progress bar

# Configuration
NUM_IMAGES = 1000
OUTPUT_DIR = "Data"
IMAGE_SIZE = (256, 256)
GAUSSIAN_SIGMA = 0.05
PSF_SIGMA = 2
PSF_SIZE = 11
VERBOSE = False  # Flag to control print statements

# Create output directories
clean_dir = Path(OUTPUT_DIR) / "clean"
noisy_dir = Path(OUTPUT_DIR) / "noisy"
clean_dir.mkdir(parents=True, exist_ok=True)
noisy_dir.mkdir(parents=True, exist_ok=True)

# PSF kernel
def gaussian_kernel(size=11, sigma=2):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

psf = gaussian_kernel(PSF_SIZE, PSF_SIGMA)

# Blur + Noise simulation
def simulate_blurred_noisy(clean_img, psf, sigma):
    blurred = convolve2d(clean_img, psf, mode='same', boundary='wrap')
    noise = np.random.normal(0, sigma, clean_img.shape)
    return np.clip(blurred + noise, 0, 1)

# Save image pair
def process_image(img_bytes, idx):
    try:
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image decode failed")
        img = cv2.resize(img, IMAGE_SIZE)
        clean_np = img / 255.0
        noisy_np = simulate_blurred_noisy(clean_np, psf, GAUSSIAN_SIGMA)

        cv2.imwrite(str(clean_dir / f"{idx:05d}.png"), (clean_np * 255).astype(np.uint8))
        cv2.imwrite(str(noisy_dir / f"{idx:05d}.png"), (noisy_np * 255).astype(np.uint8))

        if VERBOSE:
            print(f"✅ Image {idx:05d} saved")
        return True
    except Exception as e:
        if VERBOSE:
            print(f"❌ Failed to process image {idx}: {e}")
        return False

# SDSS image URLs
def download_sdss_urls(n):
    return [
        f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={180+i}&dec=0&scale=0.2&width=256&height=256"
        for i in range(n)
    ]

# Threaded download

def download_and_process_sdss():
    urls = download_sdss_urls(NUM_IMAGES)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        with tqdm(total=len(urls), desc="Simulating") as pbar:
            for idx, url in enumerate(urls):
                futures.append(executor.submit(download_process_worker, url, idx, pbar))
            for f in futures:
                f.result()
    if VERBOSE:
        print(f"\n🎉 Done! {len(futures)} images saved to `{OUTPUT_DIR}`")

def download_process_worker(url, idx, pbar):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            result = process_image(response.content, idx)
            pbar.update(1)
            return result
        else:
            if VERBOSE:
                print(f"❌ HTTP {response.status_code} at {url}")
    except Exception as e:
        if VERBOSE:
            print(f"❌ Error downloading {url}: {e}")
    pbar.update(1)
    return False

if __name__ == "__main__":
    download_and_process_sdss()