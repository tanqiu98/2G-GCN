from typing import Union

import torch
import torch.nn.functional as F


def binary_cross_entropy_loss(input, target, positive_class_weight=1, ignore_value=-1, reduction='mean'):
    """BCE loss for PyTorch tensors with optional value to be ignored.

    For the targets to be ignored (target == ignore_value), we multiply these targets and the corresponding predictions
    by zero so that we have zero loss on these terms.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    if positive_class_weight > 1:
        input = torch.where(target == 1.0, input ** positive_class_weight, input)
    criterion = F.binary_cross_entropy(input * mask, target * mask, reduction=reduction)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def budget_loss(input, target, ignore_value=-1, reduction='mean'):
    """Regularization loss to encourage a model to make the value in input zero.

    target has no effect on the loss computation and is only used to compute the valid entries in input. The keyword
    reduction also has no influence here and is kept only for signature compatibility with other PyTorch losses.
    """
    mask = (target != ignore_value).float()
    num_nonmissing_elements = mask.sum().item()
    if num_nonmissing_elements == 0:
        return torch.tensor(0.0, dtype=input.dtype, device=input.device)
    criterion = torch.mean(input * mask)
    criterion *= input.numel() / num_nonmissing_elements
    return criterion


def multi_task_loss(input: list, target: list, loss_functions: list, weight: list = None,
                    ignore_value: Union[int, float] = -1, reduction: str = 'mean'):
    """Loss function for models with multiple losses."""
    if weight is None:
        weight = [1.0] * len(input)
    criteria = []
    for input_, target_, loss_function, w in zip(input, target, loss_functions, weight):
        if loss_function is F.nll_loss:
            criterion = w * loss_function(input_, target_, ignore_index=ignore_value, reduction=reduction)
        else:
            criterion = w * loss_function(input_, target_, ignore_value=ignore_value, reduction=reduction)
        criteria.append(criterion)
    return criteria
