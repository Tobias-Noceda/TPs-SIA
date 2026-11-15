from tp5.models.denoising_autoencoder import DenoisingAutoencoder
from tp5.runners.basic import build_networks, train_run
from tp5.types.config.config import AutoencoderConfig
from tp5.types.training_run import TrainingRun

from numpy.random import RandomState

def build_denoising_autoencoder(config: AutoencoderConfig, input_size: int, random: RandomState) -> DenoisingAutoencoder:
    """Untrained denoising autoencoder described by the config (layers + noise block)."""
    noise = config["noise"]
    if noise is None:
        raise ValueError("A denoising autoencoder needs a 'noise' block in the config.")

    loss_function, derivative_loss_function = config["loss_function"]
    encoder, decoder = build_networks(config, input_size, random)

    return DenoisingAutoencoder(
        encoder,
        decoder,
        loss_function,
        derivative_loss_function,
        config["batch_size"],
        config["epsilon"],
        config["goal_pixels"],
        config["stop_at_goal"],
        random,
        noise=noise["function"],
        noise_level=noise["level"],
        copies=noise["copies"],
    )

def run_denoising(config: AutoencoderConfig) -> TrainingRun:
    """Train a denoising autoencoder: noisy glyphs as input (resampled every epoch), clean glyphs as expected output."""
    return train_run(config, build_denoising_autoencoder)
