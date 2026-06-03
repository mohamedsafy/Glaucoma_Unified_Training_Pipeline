from __future__ import annotations

from typing import Any, Iterable

try:
    from torch import Tensor
    from torch import nn
except ImportError:
    Tensor = Any

    class _Module:
        pass

    class _NN:
        Module = _Module

    nn = _NN()


class CombinedLoss(nn.Module):
    def __init__(self, weighted_losses: Iterable[tuple[Any, float]]) -> None:
        super().__init__()
        self.weighted_losses = list(weighted_losses)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # targets often come in as [B, 1, H, W], CE needs [B, H, W]
        if target.dim() == 4 and target.shape[1] == 1:
            target_sq = target.squeeze(1)
        else:
            target_sq = target

        pred_sq = pred.squeeze(1)
        loss = 0
        
        
        for fn, weight in self.weighted_losses:
            # Check if it's a standard NN loss (like CE) or an SMP loss
            if isinstance(fn, (nn.CrossEntropyLoss, nn.NLLLoss)):
                target_sq = target_sq.float()  # Ensure targets are long for CE loss
                pred_sq = pred_sq.long()  # Ensure predictions are float for loss calculations
                loss += fn(pred_sq, target_sq) * weight
            else:
                loss += fn(pred_sq, target) * weight

        return loss / len(self.weighted_losses)
