from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DataModule:
    train_ds: Any
    val_ds: Any
    test_ds: Any
    _train_loader: Any = None
    _val_loader: Any = None
    _test_loader: Any = None
    batch_size: int = 1
    num_workers: int = 0

    def train_loader(self) -> Any:
        if self._train_loader is None:
            self._train_loader = self._build_loader(self.train_ds, shuffle=True)
        return self._train_loader

    def val_loader(self) -> Any:
        if self._val_loader is None:
            self._val_loader = self._build_loader(self.val_ds, shuffle=False)
        return self._val_loader

    def test_loader(self) -> Any:
        if self._test_loader is None:
            self._test_loader = self._build_loader(self.test_ds, shuffle=False)
        return self._test_loader

    def _build_loader(self, dataset: Any, shuffle: bool) -> Any:
        try:
            from torch.utils.data import DataLoader
        except ImportError as exc:
            raise RuntimeError("Data loaders require PyTorch to be installed.") from exc

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )
