import time

from tp5.models.basic_autoencoder import BasicAutoencoder
from tp5.models.multi_layer_perceptron import MultiLayerPerceptron
from tp5.models.neuron import Neuron
from tp5.parsers.font_parser import parse_font
from tp5.types.config.config import AutoencoderConfig
from tp5.types.training_run import TrainingRun

from numpy.random import MT19937, RandomState, SeedSequence

def build_basic_autoencoder(config: AutoencoderConfig, input_size: int, random: RandomState) -> BasicAutoencoder:
    """Untrained autoencoder described by the config, its weights drawn from `random`."""
    autoencoder = config["autoencoder"]
    loss_function, derivative_loss_function = config["loss_function"]

    def network(layers, size: int) -> MultiLayerPerceptron:
        neurons = [tuple(Neuron(f, df) for f, df in layer) for layer in layers]
        return MultiLayerPerceptron(neurons, size, config["learning_rate"], random, config["initialization"])

    encoder = network(autoencoder["encoder"], input_size)
    decoder = network(autoencoder["decoder"], encoder.output_size)

    return BasicAutoencoder(
        encoder,
        decoder,
        loss_function,
        derivative_loss_function,
        config["batch_size"],
        config["epsilon"],
        config["goal_pixels"],
        config["stop_at_goal"],
        random,
    )

def run_basic(config: AutoencoderConfig) -> TrainingRun:
    """Train a basic autoencoder (expected output = input) over the glyphs of the dataset."""
    inputs, labels, shape = parse_font(config["dataset"]["path"])

    # Single generator for the whole model: same seed => same weights and same shuffling
    seed = SeedSequence(config["seed"])
    random = RandomState(MT19937(seed))

    model = build_basic_autoencoder(config, inputs.shape[1], random)

    start = time.perf_counter()
    history = model.train(inputs, inputs, config["epochs"])
    elapsed = time.perf_counter() - start

    return TrainingRun(
        model, inputs, labels, shape, history, config["goal_pixels"], int(seed.entropy), elapsed, config["source"]  # type: ignore[arg-type]
    )
