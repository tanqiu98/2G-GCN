""""Multi-task related torch code.
"""
import math
from typing import List, Optional

import torch
import torch.nn as nn


class MultiTaskLossLearner(nn.Module):
    """Module that learns contributing weights of losses in a multi-task setting.

    This multi task loss learner is an implementation of the framework proposed by Kendall et al. [1]. It works as
    an auxiliary model that learns the individual weights of the losses of an external model.

    [1] Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task Learning Using Uncertainty to Weigh Losses for
    Scene Geometry and Semantics. 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 7482–7491.
    """
    def __init__(self, loss_types: List[str], mask: Optional[List[bool]] = None):
        """Multi task loss learner.

        Arguments:
            loss_types - A list of strings specifying the type of each loss in the external model. The types can be
                one of 'mean_absolute_error', 'mean_squared_error', or 'softmax'. Abbreviations for
                'mean_absolute_error' and 'mean_squared_error' are permitted; 'mae' and 'mse', respectively.
            mask - An optional list of booleans specifying which losses to actually learn the weights. This allows
                the multi tasker module to learn the weights of a subset of losses. If None, all losses weights are
                learned.
        """
        super(MultiTaskLossLearner, self).__init__()
        self.loss_types = list(loss_types)
        self.mask = list(mask) if mask is not None else [True] * len(loss_types)

        self.log_sds = nn.Parameter(torch.zeros(len(loss_types), dtype=torch.float32))

    def forward(self, losses: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the updates values of the input losses."""
        assert len(self.loss_types) == len(losses), 'Specified loss types must match the number of input losses.'
        weighted_losses = []
        for loss_type, log_sd, loss, loss_is_learnable in zip(self.loss_types, self.log_sds, losses, self.mask):
            if loss_is_learnable:
                weighted_loss = self._compute_weighted_loss(loss, loss_type, log_sd)
                weighted_losses.append(weighted_loss)
            else:
                weighted_losses.append(loss)
        return weighted_losses

    def _compute_weighted_loss(self, loss: torch.Tensor, loss_type: str, log_sd: torch.Tensor) -> torch.Tensor:
        """Compute the updated value of a given loss."""
        if loss_type not in {'mae', 'mean_absolute_error', 'mse', 'mean_squared_error', 'softmax'}:
            raise ValueError('loss_type must be one of \'softmax\', \'mae\' or \'mse\'.')
        loss_weight = self._compute_loss_weight(loss_type, log_sd)
        weighted_loss = loss_weight * loss + log_sd
        return weighted_loss

    @staticmethod
    def _compute_loss_weight(loss_type: str, log_sd: torch.Tensor) -> torch.Tensor:
        """Compute the updated weight of a given loss."""
        if loss_type in {'mae', 'mean_absolute_error'}:
            loss_weight = math.sqrt(2.0) * torch.exp(-log_sd)
        elif loss_type in {'mse', 'mean_squared_error'}:
            loss_weight = 0.5 * torch.exp(-2 * log_sd)
        else:  # softmax
            loss_weight = torch.exp(-2 * log_sd)
        return loss_weight

    def get_weights(self) -> List[float]:
        """Return the learned weights of the losses."""
        weights = []
        for loss_type, log_sd, loss_is_learnable in zip(self.loss_types, self.log_sds, self.mask):
            if loss_is_learnable:
                weights.append(self._compute_loss_weight(loss_type, log_sd).item())
            else:
                weights.append(None)
        return weights
