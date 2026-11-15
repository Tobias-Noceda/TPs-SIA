from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tp5.interfaces.autoencoder import Autoencoder
from tp5.models.variational_autoencoder import VariationalAutoencoder
from tp5.persistence.generated import GENERATED_DIR, run_metadata, save_denoising, save_generated
from tp5.persistence.noisy_file import NOISY_DIR, list_noisy, load_noisy, save_noisy
from tp5.types.config.noise import NoiseFunctions
from tp5.types.image_shape import ImageShape
from tp5.types.samples import Samples
from tp5.types.training_run import TrainingRun

import numpy as np
import numpy.typing as npt
from numpy.random import MT19937, RandomState, SeedSequence

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
        metadata: Seed, parameters and cut conditions written next to every generated image.
        generated_dir: Where "save image" writes the generated images.
        noisy_dir: Where the noisy pattern files are listed from and saved to.
    """
    model: Autoencoder
    inputs: Samples
    labels: List[str]
    image_shape: ImageShape
    tolerance: int = 1
    ink: bool = True
    margin: float = 0.25
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_dir: str = GENERATED_DIR
    noisy_dir: str = NOISY_DIR

    latent: Samples = field(init=False, repr=False)
    binary: bool = field(init=False)
    # σ of q(z|x) per sample (variational autoencoder only)
    latent_sigma: Samples | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.inputs = np.asarray(self.inputs, dtype=np.float64)
        if self.inputs.ndim != 2 or self.inputs.shape[1] != self.image_shape.size:
            raise ValueError(f"Samples must be (N, {self.image_shape.size}) for images {tuple(self.image_shape)}, got {self.inputs.shape}.")
        if len(self.labels) != len(self.inputs):
            raise ValueError("There must be one label per sample.")

        self.latent = self.model.encode(self.inputs)
        self.binary = bool(np.isin(self.inputs, (0.0, 1.0)).all())
        if isinstance(self.model, VariationalAutoencoder):
            self.latent_sigma = np.exp(0.5 * self.model.encode_distribution(self.inputs)[1])

    @classmethod
    def from_run(cls, run: TrainingRun, generated_dir: str = GENERATED_DIR, noisy_dir: str = NOISY_DIR) -> "LatentExplorer":
        explorer = cls(
            run.model, run.inputs, run.labels, run.shape,
            metadata=run_metadata(run), generated_dir=generated_dir, noisy_dir=noisy_dir,
        )
        if run.goal_pixels is not None:
            explorer.tolerance = run.goal_pixels
        # Ink (1 = black) only for the glyphs of font.h; image datasets keep 1 = white
        explorer.ink = run.source["dataset"].get("type", "font") == "font"
        return explorer

    @property
    def variational(self) -> bool:
        return isinstance(self.model, VariationalAutoencoder)

    @property
    def latent_dim(self) -> int:
        return self.latent.shape[1]

    def bounds(self) -> npt.NDArray[np.float64]:
        """(latent_dim, 2) slider range of each coordinate: range of the training codes ± margin."""
        low, high = self.latent.min(axis=0), self.latent.max(axis=0)
        # A coordinate where every code is equal still gets a usable range
        span = np.where(high - low > 1e-9, high - low, 1.0)
        bounds = np.stack([low - self.margin * span, high + self.margin * span], axis=1)
        if self.variational:
            # Also ±3 standard deviations of the prior N(0, I): where generated samples come from
            bounds = np.stack([np.minimum(bounds[:, 0], -3.0), np.maximum(bounds[:, 1], 3.0)], axis=1)
        return bounds

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
            # Training noise of a denoising autoencoder (None for the others): defaults of the /denoise page
            "noise": self.metadata.get("noise"),
            "noise_types": [noise.value for noise in NoiseFunctions],
            "dataset": self.metadata.get("dataset"),
            # Variational autoencoder: codes are the means μ(x), σ(x) their spread; new samples come from N(0, I)
            "variational": self.variational,
            "latent_sigma": np.round(self.latent_sigma, _DECIMALS).tolist() if self.latent_sigma is not None else None,
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

    def generate(self, z: Sequence[float], name: str) -> List[Path]:
        """
        Save the decoder output for `z` into generated_dir as report figures (image + seed, parameters,
        cut conditions and coordinates): as is and, for binary datasets (the font), also binarized.
        Only called on demand ("Save image" in the page).
        """
        if len(z) != self.latent_dim:
            raise ValueError(f"Expected {self.latent_dim} latent coordinates, got {len(z)}.")
        return save_generated(
            self.model, z, name, self.image_shape, self.metadata, self.generated_dir,
            binarize=self.binary, ink=self.ink, training_inputs=self.inputs, labels=self.labels,
        )

    def sample(self, count: int, seed: int | None) -> Dict[str, Any]:
        """Latent points drawn from the prior N(0, I) (variational autoencoder) with their decoded images."""
        if not self.variational:
            raise ValueError("Sampling from the prior needs a variational autoencoder.")
        if not isinstance(count, int) or not 1 <= count <= 4096:
            raise ValueError("'count' must be an integer in [1, 4096].")
        seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        z = RandomState(MT19937(SeedSequence(seed))).standard_normal((count, self.latent_dim))
        return {"z": np.round(z, _DECIMALS).tolist(), "seed": seed, **self.decode(z)}

    # ------------------------------------------------------------------ denoising page

    def _images(self, images: Any) -> Samples:
        images = np.asarray(images, dtype=np.float64)
        if images.ndim != 2 or images.shape[1] != self.image_shape.size:
            raise ValueError(f"Expected a (images, {self.image_shape.size}) matrix, got {images.shape}.")
        if not np.isfinite(images).all():
            raise ValueError("Image values must be finite.")
        return np.clip(images, 0, 1)

    def _index(self, index: Any) -> int:
        if not isinstance(index, int) or not 0 <= index < len(self.inputs):
            raise ValueError(f"Sample index must be in [0, {len(self.inputs) - 1}].")
        return index

    def noise(self, indices: Sequence[int], noise_type: str, level: float, seed: int | None) -> Dict[str, Any]:
        """Noisy versions of the given samples, with the same noise functions used for training."""
        indices = [self._index(i) for i in indices]
        if not 0 <= float(level) <= 1:
            raise ValueError("The noise level must be in [0, 1].")
        function = NoiseFunctions(noise_type).function()
        # Small seeds: shown and typed in the page (JS numbers are exact only up to 2^53)
        seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31))
        images = function(RandomState(MT19937(SeedSequence(seed))), self.inputs[indices], float(level))
        return {"images": np.round(images, _DECIMALS).tolist(), "seed": seed}

    def reconstruct(self, images: Any) -> Dict[str, Any]:
        """Outputs of the whole autoencoder for the given images, with their latent codes and nearest samples."""
        images = self._images(images)
        latent = self.model.encode(images)
        outputs = np.clip(self.model.decode(latent), 0, 1)
        distances = self.distances(outputs)
        nearest = np.argmin(distances, axis=1)
        return {
            "outputs": np.round(outputs, _DECIMALS).tolist(),
            "latent": np.round(latent, _DECIMALS).tolist(),
            "nearest": nearest.tolist(),
            "distance": np.round(distances[np.arange(len(images)), nearest], _DECIMALS).tolist(),
        }

    def _error(self, image: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]) -> Any:
        if self.binary:
            return int(np.sum((image >= 0.5) != (reference >= 0.5)))
        return round(float(np.mean(np.abs(image - reference))), 4)

    def list_noisy(self) -> List[str]:
        return list_noisy(self.noisy_dir)

    def load_noisy(self, name: str) -> Dict[str, Any]:
        return load_noisy(self.noisy_dir, name)

    def save_noisy(self, index: int, noise: Dict[str, Any], noisy: Any) -> Path:
        """Save noisy versions of a sample (from the page, possibly edited by hand) in the noisy file format."""
        index = self._index(index)
        return save_noisy(
            self.noisy_dir, str(self.metadata.get("dataset")), self.labels[index], index, self.image_shape,
            noise, self.inputs[index], self._images(noisy),
        )

    def save_denoising(self, index: int, noisy: Any, noise: Dict[str, Any], name: str) -> Path:
        """Save the report figure original / noisy / output of a sample (named by the user)."""
        index = self._index(index)
        noisy_image = self._images([noisy])[0]
        output = np.clip(self.model.reconstruct(noisy_image[None, :])[0], 0, 1)
        original = self.inputs[index]
        distances = self.distances(output[None, :])[0]
        nearest = int(np.argmin(distances))
        info = {
            **self.metadata,
            "sample": self.labels[index],
            "index": index,
            "input_noise": noise,
            "errors": {"noisy": self._error(noisy_image, original), "output": self._error(output, original)},
            "nearest": {
                "label": self.labels[nearest],
                "distance": f"{int(distances[nearest])} wrong px" if self.binary else f"MAE {distances[nearest]:.4f}",
            },
        }
        return save_denoising(self.generated_dir, name, original, noisy_image, output, self.image_shape, info, self.ink, self.binary)
