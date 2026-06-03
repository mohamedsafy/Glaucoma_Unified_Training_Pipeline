from __future__ import annotations

from typing import Any, Optional

from new_pipeline.config import SchedulerConfig
from new_pipeline.factories.base import RegistryFactory

from torch.optim import lr_scheduler

class SchedulerFactory(RegistryFactory[SchedulerConfig]):
    registry = {'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
                'StepLR': lr_scheduler.StepLR,
                'MultiStepLR': lr_scheduler.MultiStepLR,
                'ExponentialLR': lr_scheduler.ExponentialLR,
                'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
                'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
                'LambdaLR': lr_scheduler.LambdaLR,
                'CyclicLR': lr_scheduler.CyclicLR,
                'OneCycleLR': lr_scheduler.OneCycleLR}

    @classmethod
    def create(cls, config: SchedulerConfig, optimizer: Any) -> Optional[Any]:
        if config.type is None:
            return None
        builder = cls.resolve(config.type)
        return builder(optimizer, **config.kwargs)
