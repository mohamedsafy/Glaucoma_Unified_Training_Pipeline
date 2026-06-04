from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ModelConfig:
    type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SingleLossConfig:
    type: str
    weight: float = 1.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    losses: list[SingleLossConfig] = field(default_factory=list)

    @property
    def short_desc(self) -> str:
        return "_".join([loss.type for loss in self.losses])


@dataclass
class OptimizerConfig:
    type: str
    lr: float
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    type: Optional[str] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalerConfig:
    enabled: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    train_transforms: list[Any] = field(default_factory=list)
    val_transforms: list[Any] = field(default_factory=list)
    multiplier: int = 1
    kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def short_desc(self) -> str:
        desc=''
        desc+= f"_MUL{self.multiplier}"
        desc+= '_CLAHE' if next((aug_type for aug_type in self.train_transforms if aug_type['type'] == "CLAHE" ), None) is not None else ''
        return desc



@dataclass
class DatasetConfig:
    name: str
    roi: bool = False
    local_root: Optional[str] = None
    drive_root: Optional[str] = None
    remote_source: Optional[str] = None
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    num_of_samples: int = None
    in_memory: bool = True
    kwargs: dict[str, Any] = field(default_factory=dict)
    

    @property
    def short_desc(self) -> str:
        desc=''
        desc+=self.name
        desc+= '_ROI' if self.roi else ''
        desc+= self.augmentation.short_desc
        return desc



@dataclass
class RunConfig:
    device: Any 
    epochs: int 
    img_height: int 
    img_width: int 
    accumulation_steps: int 
    model: ModelConfig 
    loss: LossConfig 
    optimizer: OptimizerConfig 
    scheduler: SchedulerConfig 
    scaler: ScalerConfig 
    dataset: DatasetConfig 
    batch_size: int = 1
    num_workers: int = 0
    exp_title: Optional[str] = None
    root_exp_dir: Optional[str] = None
    visualization_samples: Optional[Union[list[str], list[int], str]] = None
    visualization_epochs: Optional[int] = None
    
    @property
    def short_desc(self) -> str:
        desc = self.dataset.short_desc
        desc += f"_{self.model.type}_{self.optimizer.type}_{self.scheduler.type}"
        desc += f"_{self.loss.short_desc}"
        
        return desc