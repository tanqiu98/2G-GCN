from typing import Optional, Sequence, Union

import torch.nn as nn

from pyrutils.torch.general import pick_activation_function


def build_mlp(dims: Sequence[int], activations: Optional[Sequence[Union[str, dict]]] = None,
              dropout: float = 0.0, bias: bool = True):
    """Build a general Multi-layer Perceptron (MLP).

    Arguments:
        dims - An iterable containing the sequence of input/hidden/output dimensions. For instance, if
            dims = [256, 128, 64], our MLP receives input features of dimension 256, reduces it to 128, and outputs
            features of dimension 64.
        activations - An iterable containing the activations of each layer of the MLP. Each element of the iterable can
            be either a string or a dictionary. If it is a string, it specifies the name of the activation function,
            such as 'relu'; if it is a dictionary, it should contain a name key, and optional keyword arguments for the
            function. For instance, a valid input could be ['relu', {'name': 'logsoftmax', 'dim': -1}]. If activations
            is None, no activation functions are applied to the outputs of the layers of the MLP.
        dropout - Dropout probability.
        bias - Whether to include a bias term in the linear layers or not.
    Returns:
        An MLP as a PyTorch Module.
    """
    if activations is None:
        activations = ['identity'] * (len(dims) - 1)
    if len(dims) - 1 != len(activations):
        raise ValueError('Number of activations must be the same as the number of dimensions - 1.')
    layers = []
    for dim_in, dim_out, activation in zip(dims[:-1], dims[1:], activations):
        layers.append(nn.Linear(dim_in, dim_out, bias=bias))
        layers.append(pick_activation_function(activation))
        if dropout:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)
