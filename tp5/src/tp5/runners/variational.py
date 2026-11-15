from tp5.models.variational_autoencoder import VariationalAutoencoder
from tp5.runners.basic import build_networks, train_run
from tp5.types.config.config import AutoencoderConfig
from tp5.types.config.loss import LossFunctions
from tp5.types.training_run import TrainingRun

from numpy.random import RandomState

def build_variational_autoencoder(config: AutoencoderConfig, input_size: int, random: RandomState) -> VariationalAutoencoder:
    """Untrained variational autoencoder described by the config (encoder outputs μ and log Σ)."""
    beta = (config.get("variational") or {"beta": 1.0})["beta"]
    loss_function, derivative_loss_function = config["loss_function"]
    encoder, decoder = build_networks(config, input_size, random, outputs_per_latent=2)

    # mse and bce are means over the pixels, hse is a sum: the KL is scaled the same way so β = 1
    # keeps the ratio of the standard ELBO (reconstruction summed over the pixels + KL)
    loss = LossFunctions(config["source"].get("loss_function", "mse"))
    kl_weight = beta if loss == LossFunctions.HSE else beta / input_size

    return VariationalAutoencoder(
        encoder,
        decoder,
        loss_function,
        derivative_loss_function,
        config["batch_size"],
        config["epsilon"],
        config["goal_pixels"],
        config["stop_at_goal"],
        random,
        beta=beta,
        kl_weight=kl_weight,
    )

def run_variational(config: AutoencoderConfig) -> TrainingRun:
    """Train a variational autoencoder over the samples of the dataset."""
    return train_run(config, build_variational_autoencoder)
