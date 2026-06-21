from __future__ import annotations

from typing import Any, Iterable

from new_pipeline.config import OptimizerConfig
from new_pipeline.factories.base import RegistryFactory

import torch.optim as optim

class OptimizerFactory(RegistryFactory[OptimizerConfig]):
    registry = {'ADAM': optim.Adam, 'SGD': optim.SGD, 'ADAMW': torch.optim.AdamW}

    @classmethod
    def create(cls, config: OptimizerConfig, parameters: Iterable[Any]) -> Any:
        builder = cls.resolve(config.type)
        return builder(parameters, lr=config.lr, **config.kwargs)
