import os
from new_pipeline.datasets.base import StandardDataset
from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.config.run_config import AugmentationConfig
from new_pipeline.datasets.base import DatasetSourcer, DatasetSpliter
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from PIL import Image
import numpy as np

class RefugeDatasetSourcer(DatasetSourcer):
    pass

class RefugeDatasetSpliter(DatasetSpliter):
    def split(self, path: str, config: DatasetConfig, train_transforms: callable, val_transforms: callable) -> tuple[StandardDataset, StandardDataset, StandardDataset]:
        train_ds = StandardDataset(
            os.path.join(path, 'train'),
            transforms= train_transforms,
            dataset_multiplier=config.augmentation.multiplier
        )
        val_ds = StandardDataset(
            os.path.join(path, 'val'),
            transforms= val_transforms,
            n=config.num_of_samples
        )
        test_ds = StandardDataset(
            os.path.join(path, 'test'),
            transforms= val_transforms,
            n=config.num_of_samples
        )
        return train_ds, val_ds, test_ds

class RefugeStandardDataset(StandardDataset):
    pass


