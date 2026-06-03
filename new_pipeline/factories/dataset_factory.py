from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from new_pipeline.config import DatasetConfig
from new_pipeline.factories.base import RegistryFactory
from new_pipeline.datasets.base import DatasetSourcer, DatasetSpliter
from new_pipeline.datasets.refuge import RefugeDatasetSourcer, RefugeDatasetSpliter
from new_pipeline.datasets.origa import OrigaDatasetSourcer, OrigaDatasetSpliter
#from new_pipeline.datasets.drishti import DrishtiDatasetSourcer, DrishtiDatasetSpliter


class DatasetFactory(RegistryFactory[DatasetConfig]):
    registry = {
        "REFUGE": (RefugeDatasetSourcer, RefugeDatasetSpliter),
        "ORIGA": (OrigaDatasetSourcer, OrigaDatasetSpliter),
        #"DRISHTI-GS": (DrishtiDatasetSourcer, DrishtiDatasetSpliter),
    }

    @classmethod
    def create(cls, config: DatasetConfig) -> tuple[DatasetSourcer, DatasetSpliter]:
        datasource_cls, datasetspliter_cls = cls.resolve(config.name)  # Ensure the dataset name is valid

        return datasource_cls(), datasetspliter_cls()