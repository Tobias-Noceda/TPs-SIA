import enum

from tp4.types.noise_function import NoiseFunction

from tp4.functions.noise.corner import corner_noise
from tp4.functions.noise.fade import fade_noise
from tp4.functions.noise.flip import flip_noise
from tp4.functions.noise.scratch import scratch_noise
from tp4.functions.noise.stain import stain_noise

class NoiseFunctions(enum.Enum):
    FLIP = "flip"
    FADE = "fade"
    CORNER = "corner"
    SCRATCH = "scratch"
    STAIN = "stain"

    def function(self) -> NoiseFunction:
        functions: dict[NoiseFunctions, NoiseFunction] = {
            NoiseFunctions.FLIP: flip_noise,
            NoiseFunctions.FADE: fade_noise,
            NoiseFunctions.CORNER: corner_noise,
            NoiseFunctions.SCRATCH: scratch_noise,
            NoiseFunctions.STAIN: stain_noise,
        }

        return functions[self]
