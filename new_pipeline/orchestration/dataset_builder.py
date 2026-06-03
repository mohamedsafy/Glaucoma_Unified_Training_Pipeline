import os
from torch.utils.data import Dataset
from new_pipeline.config.run_config import DatasetConfig
from new_pipeline.factories.dataset_factory import DatasetFactory
from new_pipeline.factories.transform_factory import TransformFactory

class DatasetBuilder:
    def build(self, config: DatasetConfig) -> tuple[Dataset, Dataset, Dataset]:        
        # --- SOURCE ---
        datasource_cls, datasetspliter_cls = DatasetFactory.create(config)  # Ensure the dataset type is valid
        sourcer = datasource_cls
        dataset_path = sourcer.source(config)

        # --- SPLIT ---
        spliter = datasetspliter_cls
        train_transforms, val_transforms = TransformFactory.create(config.augmentation) 
        train_ds, val_ds, test_ds = spliter.split(dataset_path, config, train_transforms=train_transforms, val_transforms=val_transforms, )

        return train_ds, val_ds, test_ds
    