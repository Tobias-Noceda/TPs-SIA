import enum

from tp4.types.noise_function import NoiseFunction

from tp4.functions.noise.flip import flip_noise

class NoiseFunctions(enum.Enum):
    FLIP = "flip"

    def function(self) -> NoiseFunction:
        functions: dict[NoiseFunctions, NoiseFunction] = {
            NoiseFunctions.FLIP: flip_noise,
        }

        return functions[self]
