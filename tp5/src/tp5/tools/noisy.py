import argparse

from tp5.parsers.font_parser import parse_font
from tp5.persistence.noisy_file import NOISY_DIR, save_noisy
from tp5.types.config.noise import NoiseFunctions
from tp5.types.image_shape import ImageShape

import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

def find_sample(labels: list[str], sample: str) -> int:
    """Index of `sample` (a label of the dataset, or `#index`); raises ValueError listing the valid ones."""
    if sample.startswith("#") and sample[1:].isdigit():
        index = int(sample[1:])
        if index < len(labels):
            return index
    elif sample in labels:
        return labels.index(sample)
    raise ValueError(f"'{sample}' is not in the dataset. Valid samples: {' '.join(labels)} (or #0 … #{len(labels) - 1}).")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate noisy versions of a dataset pattern, loadable from the /denoise page.")
    parser.add_argument("-s", "--sample", required=True, help="Label of a dataset pattern (e.g. a, DEL) or #index")
    parser.add_argument("-n", "--noise", required=True, choices=[n.value for n in NoiseFunctions], help="Noise function")
    parser.add_argument("-l", "--level", type=float, required=True, help="Probability per pixel (flip, salt_pepper) or σ (gaussian), in [0, 1]")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the noise (random if omitted; the used one is stored in the file)")
    parser.add_argument("-c", "--count", type=int, default=1, help="Noisy versions to generate (default 1)")
    parser.add_argument("-d", "--dataset", default="res/datasets/font.h", help="font.h file (default res/datasets/font.h)")
    parser.add_argument("-o", "--output-dir", default=NOISY_DIR, help=f"Output directory (default {NOISY_DIR})")
    args = parser.parse_args()

    if not 0 <= args.level <= 1:
        parser.error("--level must be in [0, 1]")
    if args.count < 1:
        parser.error("--count must be at least 1")

    inputs, labels, (height, width) = parse_font(args.dataset)
    try:
        index = find_sample(labels, args.sample)
    except ValueError as error:
        parser.error(str(error))

    # Small seeds: they are shown and typed in the /denoise page (JS numbers are exact only up to 2^53)
    seed = args.seed if args.seed is not None else int(np.random.default_rng().integers(0, 2**31))
    random = RandomState(MT19937(SeedSequence(seed)))
    noise = NoiseFunctions(args.noise)
    original = inputs[index]
    noisy = noise.function()(random, np.tile(original, (args.count, 1)), args.level)

    path = save_noisy(
        args.output_dir, args.dataset, labels[index], index, ImageShape(height, width, 1),
        {"type": noise.value, "level": args.level, "seed": seed},
        original, noisy,
    )
    changed = np.sum((noisy >= 0.5) != (original >= 0.5), axis=1)
    print(f"Wrote {path}: '{labels[index]}' × {args.count} ({noise.value} {args.level}), changed pixels: {changed.tolist()}")
