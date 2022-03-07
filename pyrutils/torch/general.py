from typing import Optional, Union

import torch


def cat_valid_tensors(tensors: list, dim: int, out: Optional[torch.Tensor] = None):
    """Concatenate valid object tensors in a list.

    This function is a generalization to the torch.cat function, in which we filter the input list to remove non-tensor
    objects. Input parameters are the same as torch.cat.
    """
    return torch.cat([tensor for tensor in tensors if torch.is_tensor(tensor)], dim=dim, out=out)


def pick_activation_function(activation: Union[str, dict]):
    """Pick an activation function based on its name.

    Arguments:
        activation - Either a string or a dictionary. If a string, it specifies the function name, e.g. 'relu'. If a
            dictionary, it must contain a key named 'name' mapping to the function name, e.g. 'relu', and other
            key-value pairs in the dictionary are arguments for the function. For instance,
            {'name': 'logsoftmax', 'dim': -1}.
    Returns:
        An instantiated object of the respective PyTorch function.
    """
    name_to_fnclass = {
        'identity': torch.nn.Identity,
        'logsigmoid': torch.nn.LogSigmoid,
        'logsoftmax': torch.nn.LogSoftmax,
        'relu': torch.nn.ReLU,
        'sigmoid': torch.nn.Sigmoid,
        'softmax': torch.nn.Softmax,
        'softplus': torch.nn.Softplus,
        'tanh': torch.nn.Tanh,
    }
    try:
        activation_name = activation.get('name')
    except AttributeError:
        activation_name, kwargs = activation, {}
    else:
        del activation['name']
        kwargs = activation
    activation_obj = name_to_fnclass[activation_name.lower()](**kwargs)
    return activation_obj
