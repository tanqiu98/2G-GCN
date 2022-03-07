import torch


def sample_from_gumbel_sigmoid(probabilities: torch.Tensor, temperature: float = 1.0):
    """Sample from the Gumbel-Sigmoid distribution.

    Arg(s):
        probabilities - A tensor of shape (batch_size, 1) containing the probabilities of belonging to the positive
            class.
        temperature - Sigmoid temperature to approximate the results to the true binary distribution (temperature
            -> 0) or to smooth it out and make it uniform (temperature -> +Inf).
    Returns:
        A torch tensor of same shape as probabilities, containing the sampled probabilities for each example.
    """
    probabilities = torch.cat([probabilities, 1.0 - probabilities], dim=-1)
    g = torch.distributions.gumbel.Gumbel(0.0, 1.0).sample(probabilities.size()).to(probabilities.device)
    y = torch.log(probabilities + 1e-20) + g
    return torch.softmax(y / temperature, dim=-1)[:, :1]


def straight_through_gumbel_sigmoid(probabilities: torch.Tensor, temperature: float = 1.0, threshold: float = 0.5):
    """Straight-through estimator for binary variable using the Gumbel-Sigmoid distribution.

    Arg(s):
        probabilities - A tensor of shape (batch_size, 1) containing the probabilities of belonging to the positive
            class.
        temperature - Sigmoid temperature to approximate the results to the true binary distribution
            (temperature -> 0) or to smooth it out and make it uniform (temperature -> +Inf).
        threshold - Threshold for hard decision.
    Returns:
        Two tensors of shape (batch_size, 1) containing the estimated hard and soft probabilities, respectively.
    """
    y = sample_from_gumbel_sigmoid(probabilities, temperature=temperature)
    z = (y > threshold).float()
    z = (z - y).detach() + y
    return z, y


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold: float = 0.5):
        output = (input > threshold).float()
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = output_gradient * 1
        return grad_input


straight_through_estimator = StraightThroughEstimator.apply
