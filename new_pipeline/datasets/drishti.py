import os
from new_pipeline.datasets.base import DatasetSourcer, DatasetSpliter, StandardDataset
from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.config.run_config import AugmentationConfig
from torch.utils.data import Dataset
from torch import Tensor, from_numpy
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class DrishtiDatasetSourcer(DatasetSourcer):
    pass

class DrishtiDatasetSpliter(DatasetSpliter):
    def split(self, path: str, config: DatasetConfig, train_transforms: callable, val_transforms: callable) -> tuple[StandardDataset, StandardDataset, StandardDataset]:
        data_paths = [f for f in os.listdir(os.path.join(path, 'train/images')) if f.endswith(('.png', '.jpg', '.bmp'))]
        train_paths, val_paths = train_test_split(data_paths, test_size=0.15, random_state=42)

        train_ds = StandardDataset(
            os.path.join(path, 'train/'),
            transforms=train_transforms,
            dataset_multiplier=config.augmentation.multiplier,
            ids=train_paths,
            n=config.num_of_samples
        )
        val_ds = StandardDataset(
            os.path.join(path, 'train/'),
            transforms=val_transforms,
            ids=val_paths,
            n=config.num_of_samples
        )
        test_ds = StandardDataset(
            os.path.join(path, 'test/'),
            transforms=val_transforms,
            n=config.num_of_samples
        )
        return train_ds, val_ds, test_ds

class DrishtiStandardDataset(StandardDataset):
    pass


