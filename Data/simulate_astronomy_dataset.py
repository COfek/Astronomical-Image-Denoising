import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import numpy as np
from scipy.signal import convolve2d
from pathlib import Path

# Configuration
NUM_IMAGES = 10
OUTPUT_DIR = "Data/astro_simulated_dataset_all"
IMAGE_SIZE = (256, 256)
GAUSSIAN_SIGMA = 0.05
PSF_SIGMA = 2
PSF_SIZE = 11

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
        img = Image.open(BytesIO(img_bytes)).convert("L").resize(IMAGE_SIZE)
        clean_np = np.array(img) / 255.0
        noisy_np = simulate_blurred_noisy(clean_np, psf, GAUSSIAN_SIGMA)

        Image.fromarray((clean_np * 255).astype(np.uint8)).save(clean_dir / f"{idx:03d}.png")
        Image.fromarray((noisy_np * 255).astype(np.uint8)).save(noisy_dir / f"{idx:03d}.png")
        return True
    except Exception as e:
        print(f"❌ Failed to process image {idx}: {e}")
        return False

######################
# 1. ESA Top 100
def download_esa_images(n=10):
    base_url = "https://www.spacetelescope.org/images/archive/top100/"
    try:
        soup = BeautifulSoup(requests.get(base_url).text, "html.parser")
        img_tags = soup.select("a[href$='.jpg']")
        urls = ["https://www.spacetelescope.org" + tag["href"] for tag in img_tags][:n]
        return urls
    except Exception as e:
        print(f"❌ ESA scraping error: {e}")
        return []

######################
# 2. NASA Sample Images (predefined known public links)
def download_nasa_images(n=10):
    base_url = "https://hubblesite.org/files/live/sites/hubble/files/home/news/news-releases/"
    example_paths = [
        "2020/2020-16/2020-16a.jpg",
        "2018/2018-17/2018-17a.jpg",
        "2017/2017-04/2017-04a.jpg",
        "2015/2015-12/2015-12a.jpg",
        "2014/2014-24/2014-24a.jpg",
        "2013/2013-37/2013-37a.jpg",
        "2012/2012-37/2012-37a.jpg",
        "2010/2010-13/2010-13a.jpg",
        "2007/2007-16/2007-16a.jpg",
        "2006/2006-01/2006-01a.jpg",
    ]
    return [base_url + path for path in example_paths[:n]]

######################
# 3. SDSS API
def download_sdss_images(n=10):
    return [
        f"https://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={180+i}&dec=0&scale=0.2&width=256&height=256"
        for i in range(n)
    ]

######################
# Master Downloader
def download_and_process_all():
    sources = [
        ("ESA", download_esa_images),
        ("NASA", download_nasa_images),
        ("SDSS", download_sdss_images),
    ]

    idx = 0
    for label, url_func in sources:
        print(f"\n🔭 Downloading from {label}...")
        urls = url_func(NUM_IMAGES)
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    success = process_image(response.content, idx)
                    if success:
                        print(f"✅ [{label}] Image {idx:03d} saved")
                        idx += 1
                else:
                    print(f"❌ [{label}] HTTP {response.status_code} at {url}")
            except Exception as e:
                print(f"❌ [{label}] Error downloading {url}: {e}")

    print(f"\n🎉 Done! {idx} images saved to `{OUTPUT_DIR}`")

######################
# Run it
if __name__ == "__main__":
    download_and_process_all()
