from collections.abc import Sequence

import torch


def calculate_mean_std(batches: Sequence[torch.Tensor]):
    n_pixels = 0
    channel_sum = 0.0
    channel_sum_sq = 0.0
    for batch in batches:
        b, _, h, w = batch.shape
        n_pixels += b * h * w
        channel_sum += batch.sum(dim=[0, 2, 3])
        channel_sum_sq += (batch**2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std = (channel_sum_sq / n_pixels - mean**2).sqrt()

    return mean, std
