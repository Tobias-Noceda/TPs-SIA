from typing import List, Tuple

from tp5.parsers.font_parser import parse_font
from tp5.parsers.image_folder import parse_images
from tp5.types.config.dataset import Dataset
from tp5.types.image_shape import ImageShape
from tp5.types.samples import Samples

def load_dataset(dataset: Dataset) -> Tuple[Samples, List[str], ImageShape]:
    """Samples (flattened images in [0, 1]), labels and image shape of the dataset block of a config."""
    if dataset.get("type", "font") == "font":
        inputs, labels, (height, width) = parse_font(dataset["path"])
        # Glyphs are binary grayscale images
        return inputs, labels, ImageShape(height, width, 1)

    if "size" not in dataset:
        raise ValueError(f"Missing size for image dataset: {dataset}")
    
    height, width = dataset["size"]

    return parse_images(dataset["path"], height, width, dataset.get("channels", 3), dataset.get("fit", "stretch"))

def describe_dataset(dataset: Dataset) -> str:
    """Short description for reports and figures: path, plus size, channels and fit for image folders."""
    if dataset.get("type", "font") == "font":
        return dataset["path"]

    if "size" not in dataset:
        raise ValueError(f"Missing size for image dataset: {dataset}")
    
    height, width = dataset["size"]

    return f"{dataset['path']} ({height}x{width}x{dataset.get('channels', 3)}, {dataset.get('fit', 'stretch')})"
