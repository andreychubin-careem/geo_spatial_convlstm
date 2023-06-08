import torch
from torch import Tensor


# Losses here are modified torch losses, but without reduction, since reduction will be applied after masking

def smape(preds: Tensor, target: Tensor, epsilon: float = 1.17e-06) -> Tensor:
    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target) + torch.abs(preds), min=epsilon)
    sum_abs_per_error = 2 * torch.mean(
        torch.flatten(abs_per_error.squeeze(2), start_dim=-3, end_dim=-1), dim=-1
    ).unsqueeze(dim=1)
    return sum_abs_per_error


def mape(preds: Tensor, target: Tensor, epsilon: float = 1.17e-06) -> Tensor:
    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)
    sum_abs_per_error = torch.mean(
        torch.flatten(abs_per_error.squeeze(2), start_dim=-3, end_dim=-1), dim=-1
    ).unsqueeze(dim=1)
    return sum_abs_per_error


def mse(preds: Tensor, target: Tensor) -> Tensor:
    diff = preds - target
    sum_squared_error = torch.mean(
        torch.flatten((diff * diff).squeeze(2), start_dim=-3, end_dim=-1), dim=-1
    ).unsqueeze(dim=1)
    return sum_squared_error
