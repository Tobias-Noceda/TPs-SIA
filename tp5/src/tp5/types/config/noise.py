import enum

from typing import TypedDict

from tp5.types.noise_function import NoiseFunction

from tp5.functions.noise.flip import flip_noise
from tp5.functions.noise.gaussian import gaussian_noise
from tp5.functions.noise.salt_pepper import salt_pepper_noise

class NoiseFunctions(enum.Enum):
    FLIP = "flip"
    SALT_PEPPER = "salt_pepper"
    GAUSSIAN = "gaussian"

    def function(self) -> NoiseFunction:
        functions: dict[NoiseFunctions, NoiseFunction] = {
            NoiseFunctions.FLIP: flip_noise,
            NoiseFunctions.SALT_PEPPER: salt_pepper_noise,
            NoiseFunctions.GAUSSIAN: gaussian_noise,
        }

        return functions[self]

class NoiseDefinition(TypedDict):
    """
    Noise of a denoising autoencoder.

    Attributes:
        type: Which noise.
        function: The noise function.
        level: Probability per pixel (flip, salt_pepper) or standard deviation (gaussian).
        copies: Noisy versions of every pattern per epoch.
    """
    type: NoiseFunctions
    function: NoiseFunction
    level: float
    copies: int
