from __future__ import annotations

from typing import Any

from new_pipeline.config import LossConfig, SingleLossConfig
from new_pipeline.factories.base import RegistryFactory
from new_pipeline.runtime.combined_loss import CombinedLoss

import segmentation_models_pytorch as smp
import torch.nn as nn

class LossFactory(RegistryFactory[SingleLossConfig]):
    registry = {"BCE": nn.BCELoss,
                "CE": nn.CrossEntropyLoss,
                "DICE": smp.losses.DiceLoss}  

    @classmethod
    def create(cls, config: LossConfig) -> CombinedLoss:
        weighted_losses: list[tuple[Any, float]] = []
        for loss_config in config.losses:
            builder = cls.resolve(loss_config.type)
            try:
                weighted_losses.append((builder(**loss_config.kwargs), loss_config.weight))
            except Exception as e:
                raise Exception(f"Error creating loss function for type {loss_config.type}: {e}")
        return CombinedLoss(weighted_losses)
