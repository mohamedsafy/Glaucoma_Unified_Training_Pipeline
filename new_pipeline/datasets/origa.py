import os
from new_pipeline.datasets.base import DatasetSourcer, DatasetSpliter, StandardDataset
from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.config.run_config import AugmentationConfig
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class OrigaDatasetSourcer(DatasetSourcer):
    pass

class OrigaDatasetSpliter(DatasetSpliter):
    def split(self, path: str, config: DatasetConfig, train_transforms: callable, val_transforms: callable) -> tuple[StandardDataset, StandardDataset, StandardDataset]:
        data_paths = [f for f in os.listdir(os.path.join(path, 'images')) if f.endswith(('.png', '.jpg', '.bmp'))]
        train_val_paths, test_paths = train_test_split(data_paths, test_size=0.15, random_state=42)
        train_paths, val_paths = train_test_split(train_val_paths, test_size=0.15/(1-0.15), random_state=42)

        train_ds = StandardDataset(
            os.path.join(path),
            transforms=train_transforms,
            dataset_multiplier=config.augmentation.multiplier,
            ids=train_paths,
            n=config.num_of_samples,
            in_memory=config.in_memory,
        )
        val_ds = StandardDataset(
            os.path.join(path),
            transforms=val_transforms,
            ids=val_paths,
            n=config.num_of_samples,
            in_memory=config.in_memory,
        )
        test_ds = StandardDataset(
            os.path.join(path),
            transforms=val_transforms,
            ids=test_paths,
            n=config.num_of_samples,
            in_memory=config.in_memory,
        )
        return train_ds, val_ds, test_ds

class OrigaStandardDataset(StandardDataset):
    pass


