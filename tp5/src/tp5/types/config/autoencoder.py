import enum

from typing import List, Tuple, TypedDict

from tp5.types.activation_function import ActivationFunction, DerivativeActivationFunction

class AutoencoderType(enum.Enum):
    BASIC = "basic"
    DENOISING = "denoising"
    VARIATIONAL = "variational"

# Activation and derivative of each neuron of a layer
type LayerDefinition = List[Tuple[ActivationFunction, DerivativeActivationFunction]]

class AutoencoderDefinition(TypedDict):
    """
    Definition of the autoencoder model.

    Attributes:
        type: Kind of autoencoder, decides what main runs.
        encoder: Layers of the encoder (the input layer is not written; the last one is the latent space).
        decoder: Layers of the decoder (its input is the latent space; the last one must match the input size).
    """
    type: AutoencoderType
    encoder: List[LayerDefinition]
    decoder: List[LayerDefinition]
