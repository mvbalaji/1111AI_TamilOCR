"""
augment.py — image degradation pipeline for P2 synthetic data.

Applies realistic document degradations to PIL images.  All transforms are
deterministic given a seed so the manifest can reproduce any image.

Degradation types and typical parameter ranges:
  gaussian_blur  : sigma 0.5–2.0  (out-of-focus, scan softness)
  jpeg_compress  : quality 40–85  (scan-to-PDF artifacts)
  add_noise      : amount 5–25    (scanner noise, photocopy grain)
  random_skew    : degrees ±0–3   (page not flat on scanner)
  brightness     : factor 0.7–1.3 (lighting variation)
  salt_pepper    : density 0.002–0.01

Usage:
  from augment import augment_image
  aug_img = augment_image(img, seed=42, level="medium")

  from augment import augment_pipeline
  aug_img = augment_pipeline(img, transforms=["blur","jpeg","noise"], seed=0)
"""

from __future__ import annotations

import io
import math
import random
from typing import Literal

try:
    from PIL import Image, ImageFilter, ImageEnhance
    import numpy as np
except ImportError as e:
    raise ImportError("Pillow and numpy required: pip install Pillow numpy") from e


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

def gaussian_blur(img: Image.Image, sigma: float = 1.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def jpeg_compress(img: Image.Image, quality: int = 70) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def add_noise(img: Image.Image, amount: float = 15.0, rng: random.Random | None = None) -> Image.Image:
    arr = np.array(img.convert("RGB")).astype(np.float32)
    if rng is None:
        rng = random.Random()
    seed = rng.randint(0, 2**31)
    np.random.seed(seed)
    noise = np.random.normal(0, amount, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    result = Image.fromarray(arr)
    return result.convert(img.mode) if img.mode != "RGB" else result


def random_skew(img: Image.Image, max_degrees: float = 2.0, rng: random.Random | None = None) -> Image.Image:
    if rng is None:
        rng = random.Random()
    angle = rng.uniform(-max_degrees, max_degrees)
    return img.rotate(angle, expand=False, fillcolor=255 if img.mode == "L" else (255, 255, 255))


def brightness_jitter(img: Image.Image, factor: float = 1.0) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def salt_pepper(img: Image.Image, density: float = 0.005, rng: random.Random | None = None) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    if rng is None:
        rng = random.Random()
    seed = rng.randint(0, 2**31)
    np.random.seed(seed)
    mask = np.random.random(arr.shape[:2])
    arr[mask < density / 2] = 0
    arr[mask > 1 - density / 2] = 255
    result = Image.fromarray(arr)
    return result.convert(img.mode) if img.mode != "RGB" else result


# ---------------------------------------------------------------------------
# Level presets
# ---------------------------------------------------------------------------

LEVEL_PRESETS: dict[str, dict] = {
    "clean": {},   # no augmentation
    "light": {
        "blur":       {"sigma": 0.5},
        "brightness": {"factor_range": (0.9, 1.1)},
    },
    "medium": {
        "blur":       {"sigma": 1.0},
        "jpeg":       {"quality_range": (65, 85)},
        "noise":      {"amount": 10.0},
        "skew":       {"max_degrees": 1.5},
        "brightness": {"factor_range": (0.8, 1.2)},
    },
    "heavy": {
        "blur":       {"sigma": 1.8},
        "jpeg":       {"quality_range": (40, 65)},
        "noise":      {"amount": 20.0},
        "skew":       {"max_degrees": 3.0},
        "brightness": {"factor_range": (0.7, 1.3)},
        "salt_pepper": {"density": 0.006},
    },
}


def augment_image(
    img: Image.Image,
    seed: int = 0,
    level: Literal["clean", "light", "medium", "heavy"] = "medium",
) -> Image.Image:
    """Apply a random combination of degradations at the given level."""
    preset = LEVEL_PRESETS[level]
    if not preset:
        return img

    rng = random.Random(seed)

    if "blur" in preset and rng.random() < 0.6:
        img = gaussian_blur(img, sigma=preset["blur"]["sigma"])

    if "jpeg" in preset and rng.random() < 0.5:
        lo, hi = preset["jpeg"]["quality_range"]
        img = jpeg_compress(img, quality=rng.randint(lo, hi))

    if "noise" in preset and rng.random() < 0.4:
        img = add_noise(img, amount=preset["noise"]["amount"], rng=rng)

    if "skew" in preset and rng.random() < 0.3:
        img = random_skew(img, max_degrees=preset["skew"]["max_degrees"], rng=rng)

    if "brightness" in preset and rng.random() < 0.5:
        lo, hi = preset["brightness"]["factor_range"]
        img = brightness_jitter(img, factor=rng.uniform(lo, hi))

    if "salt_pepper" in preset and rng.random() < 0.2:
        img = salt_pepper(img, density=preset["salt_pepper"]["density"], rng=rng)

    return img


def augment_pipeline(
    img: Image.Image,
    transforms: list[str],
    seed: int = 0,
    params: dict | None = None,
) -> Image.Image:
    """Apply a specific ordered list of transforms with optional param overrides."""
    rng = random.Random(seed)
    params = params or {}
    for t in transforms:
        if t == "blur":
            img = gaussian_blur(img, **params.get("blur", {"sigma": 1.0}))
        elif t == "jpeg":
            img = jpeg_compress(img, **params.get("jpeg", {"quality": 70}))
        elif t == "noise":
            img = add_noise(img, rng=rng, **params.get("noise", {"amount": 10.0}))
        elif t == "skew":
            img = random_skew(img, rng=rng, **params.get("skew", {"max_degrees": 2.0}))
        elif t == "brightness":
            img = brightness_jitter(img, **params.get("brightness", {"factor": 0.9}))
        elif t == "salt_pepper":
            img = salt_pepper(img, rng=rng, **params.get("salt_pepper", {"density": 0.005}))
    return img


if __name__ == "__main__":
    # Quick smoke test — renders a white rectangle and applies each level
    img = Image.new("RGB", (400, 60), color=(255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "Test augmentation", fill=(0, 0, 0))
    for level in ["clean", "light", "medium", "heavy"]:
        out = augment_image(img.copy(), seed=42, level=level)
        print(f"  {level:8s}: mode={out.mode} size={out.size} OK")
    print("augment.py smoke test passed.")
