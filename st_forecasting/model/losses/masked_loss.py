import torch
from torch import Tensor
from typing import Callable


def masked_loss(preds: Tensor, target: Tensor, mask: Tensor, loss_fn: Callable) -> Tensor:
    """
    !!! Might create problems if some examples have no 'empty' tiles.
    :param preds:
    :param target:
    :param mask:
    :param loss_fn:
    :return:
    """
    assert preds.shape == target.shape, 'preds and target shapes do not match'
    assert len(mask.shape) == 5, 'something wrong with mask shape'

    _, n_steps, f, w, h = target.shape
    preds_masked = preds * mask
    mask_inv = torch.where(mask == 0.0, 1.0, 0.0)
    zeros_count = torch.flatten(mask_inv, start_dim=-3, end_dim=-1).sum(2)
    # zeros_count = torch.where(zeros_count == 0.0, 1.0, zeros_count)
    loss = loss_fn(preds_masked, target)
    loss = (loss * w * h) / zeros_count

    return loss.mean()
