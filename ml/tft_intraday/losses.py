"""One quantile loss per target, each over only the rows where its target is valid."""

from __future__ import annotations

import torch
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss


def bit_weights(weight: torch.Tensor, n: int) -> list[torch.Tensor]:
    """Per target i: bit i of the panel's tmask, rescaled so the weights average 1 over the
    valid rows of the batch. A batch with no valid row for a target weighs it 0."""
    bits = weight.long()
    out = []
    for i in range(n):
        w = ((bits >> i) & 1).to(torch.float32)
        out.append(w * (w.numel() / w.sum().clamp(min=1.0)))
    return out


class MaskedMultiLoss(MultiLoss):
    """MultiLoss whose dataset weight is a validity bitmask (bit i = target i), not a weight
    shared by every target. The masked mean keeps a rarely valid target (rest of day on a
    partial day) from counting less than the others."""

    def __init__(self, quantiles: list[float], n_targets: int):
        # separate instances: [QuantileLoss()] * n would share one metric state
        # (no attributes of its own: MultiLoss reads e.g. `.quantiles` from its metrics)
        super().__init__([QuantileLoss(quantiles=list(quantiles)) for _ in range(n_targets)])

    def update(self, y_pred, y_actual, **kwargs) -> None:
        target, weight = y_actual
        for idx, (metric, w) in enumerate(zip(self.metrics, bit_weights(weight, len(self)))):
            metric.update(y_pred[idx], (target[idx], w))

    def forward(self, y_pred, y_actual, **kwargs):
        target, weight = y_actual
        res = [metric(y_pred[idx], (target[idx], w)) * self.weights[idx]
               for idx, (metric, w) in enumerate(zip(self.metrics, bit_weights(weight, len(self))))]
        return torch.stack(res, dim=0).sum(0)
