from dataclasses import dataclass, field
from typing import List, Tuple, override

from tp5.models.basic_autoencoder import BasicAutoencoder
from tp5.types.float import Float
from tp5.types.samples import Samples

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState

# log Σ is clipped to this range before exp(): keeps σ and the KL finite while training
_LOGVAR_LIMIT = 10.0

# Variational autoencoder: the encoder q_φ(z|x) outputs the mean μ(x) and the log-variance log Σ(x)
# of a diagonal Gaussian, z is sampled with the reparametrization trick z = μ + ε ⊙ σ (ε ~ N(0, I)),
# and the decoder p_θ(x|z) reconstructs x. Loss J = reconstruction + λ KL(q(z|x) || N(0, I)), with
# KL = ½ Σ_k (exp(log Σ) + μ² − 1 − log Σ).
# @link https://arxiv.org/abs/1312.6114
@dataclass
class VariationalAutoencoder(BasicAutoencoder):
    """
    Requirements: the encoder's last layer has 2·k linear neurons (first k = μ, last k = log Σ) and
    the decoder takes k inputs.

    Attributes:
        beta: λ of J = L + λ R as written in the config (1 = standard ELBO).
        kl_weight: Weight actually applied to the KL of each sample. The reconstruction losses are
            means over the pixels, so the builder sets beta / pixels: keeps the ratio between both
            terms of the standard ELBO (sum over pixels + KL) without changing the losses.
        loss_terms: (reconstruction, mean KL) measured at the end of every epoch.
    """
    beta: float = field(default=1.0, kw_only=True)
    kl_weight: float = field(default=1.0, kw_only=True)
    loss_terms: List[Tuple[float, float]] = field(default_factory=list, kw_only=True, repr=False)

    @override
    def _check_sizes(self) -> None:
        if self.encoder.output_size != 2 * self.decoder.input_size:
            raise ValueError(
                f"The encoder must output 2 values (μ, log Σ) per latent dimension: it outputs "
                f"{self.encoder.output_size} but the decoder takes {self.decoder.input_size} (expected {2 * self.decoder.input_size})."
            )
        if self.decoder.output_size != self.encoder.input_size:
            raise ValueError(
                f"The decoder outputs {self.decoder.output_size} values but the encoder expects {self.encoder.input_size}."
            )

    def _split(self, encoded: Samples) -> Tuple[Samples, Samples]:
        k = self.latent_size
        return encoded[:, :k], encoded[:, k:]

    def encode_distribution(self, inputs: Samples) -> Tuple[Samples, Samples]:
        """(μ, log Σ) of q(z|x) for each input."""
        return self._split(self.encoder.predict(self._validate(inputs, self.encoder.input_size)))

    @override
    def encode(self, inputs: Samples) -> Samples:
        # Deterministic code: the mean of q(z|x) (latent map, explorer, reconstructions)
        return self.encode_distribution(inputs)[0]

    def sample(self, count: int, random: RandomState) -> Tuple[Samples, Samples]:
        """Generate `count` new samples from the prior: z ~ N(0, I), x = decode(z). Returns (z, x)."""
        z = random.standard_normal((count, self.latent_size))
        return z, self.decode(z)

    @staticmethod
    def kl_divergence(mu: Samples, logvar: Samples) -> npt.NDArray[np.float64]:
        """KL(N(μ, Σ) || N(0, I)) per sample, Σ diagonal given as log Σ."""
        logvar = np.clip(logvar, -_LOGVAR_LIMIT, _LOGVAR_LIMIT)
        return 0.5 * np.sum(np.exp(logvar) + mu ** 2 - 1 - logvar, axis=1)

    @override
    def _train_batch(self, inputs: Samples, expected: Samples, previous_loss: Float) -> None:
        mu, raw_logvar = self._split(self.encoder.forward(inputs))
        logvar = np.clip(raw_logvar, -_LOGVAR_LIMIT, _LOGVAR_LIMIT)
        sigma = np.exp(0.5 * logvar)

        # Reparametrization trick: the randomness is an extra input ε, so z is differentiable in μ and σ
        epsilon = self.random.standard_normal(mu.shape)
        z = mu + epsilon * sigma

        output = self.decoder.forward(z)
        # The decoder only gets the reconstruction gradient; backward returns ∂L/∂z
        gradient_z = self.decoder.backward(self.derivative_loss_function(output, expected))

        # Encoder: reconstruction (through z) + λ · KL, with ∂z/∂μ = 1 and ∂z/∂log Σ = ε σ / 2
        gradient_mu = gradient_z + self.kl_weight * mu
        gradient_logvar = gradient_z * epsilon * sigma / 2 + self.kl_weight * 0.5 * (np.exp(logvar) - 1)
        # Outside the clip range log Σ is constant: no gradient
        gradient_logvar = np.where(np.abs(raw_logvar) <= _LOGVAR_LIMIT, gradient_logvar, 0.0)
        self.encoder.backward(np.concatenate([gradient_mu, gradient_logvar], axis=1))

        self.decoder.step(previous_loss)
        self.encoder.step(previous_loss)

    @override
    def _epoch_loss(self, inputs: Samples, expected: Samples, reconstructions: Samples) -> Float:
        # Deterministic (z = μ, no ε): the stop criterion does not depend on the random draw
        reconstruction = float(self.loss_function(reconstructions, expected))
        kl = float(np.mean(self.kl_divergence(*self.encode_distribution(inputs))))
        self.loss_terms.append((reconstruction, kl))
        return np.float64(reconstruction + self.kl_weight * kl)
