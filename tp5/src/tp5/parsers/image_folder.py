from pathlib import Path
from typing import List, Literal, Tuple

from tp5.types.image_shape import ImageShape
from tp5.types.samples import Samples

import numpy as np
from PIL import Image

# File types read from an image folder
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

# How an image of another aspect ratio becomes (height, width):
# crop = central region with the target aspect ratio (fills, loses the borders: portraits);
# pad = whole image centered on white (keeps everything: flags of different proportions);
# stretch = resized ignoring the aspect ratio.
type Fit = Literal["crop", "pad", "stretch"]

def image_files(folder: str | Path) -> List[Path]:
    """Image files under `folder` (recursively), sorted by path."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

def fit_image(image: Image.Image, height: int, width: int, fit: Fit = "stretch") -> Image.Image:
    """Resize `image` to width x height following `fit` (white background for transparency and padding)."""
    if image.mode in ("RGBA", "LA", "P"):
        # Transparent pixels on white, not black
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")

    if fit == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)

    target = width / height
    current = image.width / image.height
    if fit == "crop":
        if current > target:
            new_width = round(image.height * target)
            left = (image.width - new_width) // 2
            image = image.crop((left, 0, left + new_width, image.height))
        else:
            new_height = round(image.width / target)
            top = (image.height - new_height) // 2
            image = image.crop((0, top, image.width, top + new_height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    if fit == "pad":
        scale = min(width / image.width, height / image.height)
        resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
        return canvas

    raise ValueError(f"Unknown fit '{fit}' (crop, pad or stretch).")

def parse_images(folder: str, height: int, width: int, channels: int = 3, fit: Fit = "stretch") -> Tuple[Samples, List[str], ImageShape]:
    """
    Read every image under `folder` as a sample.

    Returns:
        Samples (images, height · width · channels) flattened row-major and channels-last with values
        in [0, 1], their labels (path relative to `folder` without extension, e.g. `homer_simpson/pic_0012`)
        and the image shape.
    """
    if channels not in (1, 3):
        raise ValueError("Images must have 1 (grayscale) or 3 (RGB) channels.")
    files = image_files(folder)
    if not files:
        raise ValueError(f"No images ({', '.join(IMAGE_SUFFIXES)}) found in {folder}.")

    samples = []
    for path in files:
        with Image.open(path) as image:
            fitted = fit_image(image, height, width, fit)
        if channels == 1:
            fitted = fitted.convert("L")
        samples.append(np.asarray(fitted, dtype=np.float64).reshape(-1) / 255.0)

    labels = [path.relative_to(folder).with_suffix("").as_posix() for path in files]
    return np.array(samples), labels, ImageShape(height, width, channels)
