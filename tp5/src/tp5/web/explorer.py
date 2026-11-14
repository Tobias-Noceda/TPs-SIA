from dataclasses import dataclass, field
from typing import Any, Dict, List

from tp5.interfaces.autoencoder import Autoencoder
from tp5.types.image_shape import ImageShape
from tp5.types.samples import Samples
from tp5.types.training_run import TrainingRun

import numpy as np
import numpy.typing as npt

# Decimals sent to the browser: enough for sliders and images, keeps the JSON small
_DECIMALS = 5

@dataclass
class LatentExplorer:
    """
    What the web explorer needs from a trained autoencoder, independent of the dataset: the model,
    the training samples (flattened images, values in [0, 1]) with their labels and the image shape.

    Attributes:
        model: Trained autoencoder (only encode / decode are used: works for any implementation).
        inputs: Training samples (N, height · width · channels).
        labels: Name of each sample.
        image_shape: Shape of the images, decides how the page draws them.
        tolerance: For binary datasets, a decoded image farther than this (wrong pixels) from every
            sample is considered new; ignored for continuous datasets.
        ink: Grayscale images are drawn with 1 = black (glyphs) instead of 1 = white.
        margin: Fraction of each latent coordinate range added on both sides of the slider bounds.
    """
    model: Autoencoder
    inputs: Samples
    labels: List[str]
    image_shape: ImageShape
    tolerance: int = 1
    ink: bool = True
    margin: float = 0.25

    latent: Samples = field(init=False, repr=False)
    binary: bool = field(init=False)

    def __post_init__(self) -> None:
        self.inputs = np.asarray(self.inputs, dtype=np.float64)
        if self.inputs.ndim != 2 or self.inputs.shape[1] != self.image_shape.size:
            raise ValueError(f"Samples must be (N, {self.image_shape.size}) for images {tuple(self.image_shape)}, got {self.inputs.shape}.")
        if len(self.labels) != len(self.inputs):
            raise ValueError("There must be one label per sample.")

        self.latent = self.model.encode(self.inputs)
        self.binary = bool(np.isin(self.inputs, (0.0, 1.0)).all())

    @classmethod
    def from_run(cls, run: TrainingRun) -> "LatentExplorer":
        return cls(run.model, run.inputs, run.labels, run.shape, tolerance=run.goal_pixels)

    @property
    def latent_dim(self) -> int:
        return self.latent.shape[1]

    def bounds(self) -> npt.NDArray[np.float64]:
        """(latent_dim, 2) slider range of each coordinate: range of the training codes ± margin."""
        low, high = self.latent.min(axis=0), self.latent.max(axis=0)
        # A coordinate where every code is equal still gets a usable range
        span = np.where(high - low > 1e-9, high - low, 1.0)
        return np.stack([low - self.margin * span, high + self.margin * span], axis=1)

    def dataset(self) -> Dict[str, Any]:
        """Everything the page loads once: samples, their latent codes and how to draw them."""
        return {
            "labels": self.labels,
            "images": np.round(self.inputs, _DECIMALS).tolist(),
            "latent": np.round(self.latent, _DECIMALS).tolist(),
            "reconstructions": np.round(np.clip(self.model.decode(self.latent), 0, 1), _DECIMALS).tolist(),
            "image_shape": {"height": self.image_shape.height, "width": self.image_shape.width, "channels": self.image_shape.channels},
            "latent_dim": self.latent_dim,
            "bounds": np.round(self.bounds(), _DECIMALS).tolist(),
            "binary": self.binary,
            "tolerance": self.tolerance if self.binary else None,
            "ink": self.ink,
        }

    def distances(self, images: Samples) -> npt.NDArray[np.float64]:
        """
        (images, samples) distance from each image to each training sample: wrong pixels after
        binarizing at 0.5 for binary datasets, mean absolute error otherwise.
        """
        if self.binary:
            return np.sum((images[:, None, :] >= 0.5) != (self.inputs[None, :, :] >= 0.5), axis=2).astype(np.float64)
        return np.mean(np.abs(images[:, None, :] - self.inputs[None, :, :]), axis=2)

    def decode(self, latent: Samples) -> Dict[str, Any]:
        """Decode a batch of latent points, with the nearest training sample of each output."""
        latent = np.asarray(latent, dtype=np.float64)
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"Expected a (points, {self.latent_dim}) matrix, got {latent.shape}.")
        if not np.isfinite(latent).all():
            raise ValueError("Latent points must be finite.")

        images = np.clip(self.model.decode(latent), 0, 1)
        distances = self.distances(images)
        nearest = np.argmin(distances, axis=1)
        return {
            "images": np.round(images, _DECIMALS).tolist(),
            "nearest": nearest.tolist(),
            "distance": np.round(distances[np.arange(len(images)), nearest], _DECIMALS).tolist(),
        }
