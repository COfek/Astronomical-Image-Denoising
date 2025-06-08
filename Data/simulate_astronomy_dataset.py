import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple
import os

import cv2
import numpy as np
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from scipy.signal import convolve2d
from tqdm import tqdm
from threading import Lock

# === Configuration ===
NUM_IMAGES: int = 100000 # Number of images to simulate
OUTPUT_DIR: str = "Data" # Output directory for clean and noisy images
IMAGE_SIZE: Tuple[int, int] = (256, 256)  # Size to resize images to
GAUSSIAN_SIGMA: float = 0.05 # Standard deviation for Gaussian noise
PSF_SIGMA: float = 2 # Standard deviation for Gaussian PSF
PSF_SIZE: int = 11 # Size of the Gaussian PSF (must be odd)
VERBOSE: bool = False  # Flag to control print statements

# === Output Directories ===
clean_dir: Path = Path(OUTPUT_DIR) / "clean"
noisy_dir: Path = Path(OUTPUT_DIR) / "noisy"
clean_dir.mkdir(parents=True, exist_ok=True)
noisy_dir.mkdir(parents=True, exist_ok=True)

def gaussian_kernel(size: int = 11, sigma: float = 2) -> np.ndarray:
    """
    Generate a normalized 2D Gaussian kernel.

    Args:
        size (int): Width and height of the kernel (must be odd).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: 2D Gaussian kernel normalized to sum to 1.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

psf: np.ndarray = gaussian_kernel(PSF_SIZE, PSF_SIGMA)
np.save(Path(OUTPUT_DIR) / "psf.npy", psf)

def simulate_blurred_noisy(clean_img: np.ndarray, psf: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur and additive noise to a clean image.

    Args:
        clean_img (np.ndarray): Clean input image, normalized to [0, 1].
        psf (np.ndarray): Point spread function for blurring.
        sigma (float): Standard deviation of Gaussian noise.

    Returns:
        np.ndarray: Simulated noisy and blurred image.
    """
    blurred = convolve2d(clean_img, psf, mode='same', boundary='wrap')
    noise = np.random.normal(0, sigma, clean_img.shape)
    return np.clip(blurred + noise, 0, 1)

def process_image(img_bytes: bytes, idx: int) -> bool:
    """
    Process an image: decode, resize, normalize, simulate noise+blur, and save.

    Args:
        img_bytes (bytes): Raw image content as byte stream.
        idx (int): Index for filename.

    Returns:
        bool: True if processing and saving succeeded, False otherwise.
    """
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

def download_sdss_urls(n: int) -> List[str]:
    """
    Generate SDSS image URLs.

    Args:
        n (int): Number of URLs to generate.

    Returns:
        List[str]: List of SDSS image URLs.
    """
    return [
        f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={180+i}&dec=0&scale=0.2&width=256&height=256"
        for i in range(n)
    ]

def download_process_worker(url: str, idx: int, pbar: tqdm) -> bool:
    """
    Download and process a single image from a URL.

    Args:
        url (str): URL to download the image from.
        idx (int): Image index for saving.
        pbar (tqdm): Progress bar to update after each attempt.

    Returns:
        bool: True if successful, False otherwise.
    """
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

def download_and_process_sdss() -> None:
    """
    Download images from SDSS and process them using multi-threading.
    Saves clean and noisy pairs to output directories.
    """
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

#  === Downsample DIV2K Dataset === 

def downsample_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Downsamples an image by a given factor using area interpolation."""
    h, w = img.shape[:2]
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)

def process_and_save(img_path: Path, downsampled_dir: Path, factor: int, pbar: tqdm, lock: Lock) -> None:
    """Reads, downsamples, and saves the high-res and low-res versions of an image."""
    img = cv2.imread(str(img_path))
    if img is not None:
        down = downsample_image(img, factor)
        cv2.imwrite(str(downsampled_dir / img_path.name), down)
    with lock:
        pbar.update(1)

def download_and_process_div2k(factor: int = 4) -> None:
    """Download the DIV2K dataset from Kaggle, extract it, and downsample the images."""
    
    # === Setup directories ===
    data_dir = Path("Data")
    downsampled_dir = data_dir / "downsampled"
    extract_dir = data_dir/ "div2k_raw"
    zip_path = extract_dir / "div2k-dataset.zip"
    downsampled_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # === Download with Kaggle API ===
    if not zip_path.exists():
        print("🔽 Downloading DIV2K dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("joe1995/div2k-dataset", path=extract_dir, unzip=False)
    else:
        print("📁 Found existing ZIP file. Skipping download.")

    # === Unzip dataset ===
    print("📦 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # === Process images in parallel with progress bar ===
    image_folder = extract_dir / "DIV2K_train_HR" / "DIV2K_train_HR"
    image_files = list(image_folder.glob("*.png"))

    if not image_files:
        raise RuntimeError(f"No PNG files found in {image_folder}")

    print(f"📸 Processing {len(image_files)} images with {os.cpu_count()} threads...")

    lock = Lock()
    with tqdm(total=len(image_files), desc="🔧 Processing") as pbar:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            for img_path in image_files:
                executor.submit(process_and_save, img_path, downsampled_dir, factor, pbar, lock)

    print("✅ Done. Saved to `Data/highres` and `Data/downsampled`.")

