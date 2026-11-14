from typing import Any, Dict, Tuple, TypedDict

from tp5.types.config.autoencoder import AutoencoderDefinition
from tp5.types.config.dataset import Dataset
from tp5.types.float import Float
from tp5.types.initialization_function import InitializationFunction
from tp5.types.learning_rate_function import LearningRateFactory
from tp5.types.loss_function import DerivativeLossFunction, LossFunction

class AutoencoderConfig(TypedDict):
    """
    Configuration for the autoencoder model.

    Attributes:
        autoencoder: Type and layers of the encoder and decoder.
        loss_function: Loss and its derivative with respect to the outputs.
        learning_rate: Factory of the learning rate function (one per parameter array).
        epochs: Maximum number of training epochs.
        batch_size: Samples per weight update (1 = online, dataset size = batch).
        epsilon: Training stops once the epoch loss is below it.
        goal_pixels: A pattern is learned with at most this many wrong pixels.
        stop_at_goal: Training stops once every pattern is learned.
        seed: Seed for weight initialization and shuffling (None = random).
        initialization: Draws the initial weights and biases of each layer.
        dataset: Where the patterns are read from.
        source: The validated config as written (JSON-serializable), stored with the results.
    """
    autoencoder: AutoencoderDefinition
    loss_function: Tuple[LossFunction, DerivativeLossFunction]
    learning_rate: LearningRateFactory
    epochs: int
    batch_size: int
    epsilon: Float
    goal_pixels: int
    stop_at_goal: bool
    seed: int | None
    initialization: InitializationFunction
    dataset: Dataset
    source: Dict[str, Any]
