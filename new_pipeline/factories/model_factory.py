from __future__ import annotations

from typing import Any

from new_pipeline.config import ModelConfig
from new_pipeline.factories.base import RegistryFactory
from new_pipeline.factories.dummy_segmentation_model import DummySegmentationModel

import segmentation_models_pytorch as smp

def build_dummy_model(**kwargs):
        return DummySegmentationModel(**kwargs)

def build_efficientunet_b7(**kwargs):
        return smp.UnetPlusPlus(encoder_name='efficientnet-b7', encoder_weights='imagenet', in_channels=3, classes=3)

class ModelFactory(RegistryFactory[ModelConfig]):
    registry = { 'dummy' : build_dummy_model,
                    'efficientunet-b7': build_efficientunet_b7}

    @classmethod
    def create(cls, config: ModelConfig) -> Any:
        builder = cls.resolve(config.type)
        return builder(**config.kwargs)

    
