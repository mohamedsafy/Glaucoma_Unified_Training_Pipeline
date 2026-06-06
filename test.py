import os
import time

import torch
from torch.utils.data import DataLoader

from new_pipeline.config.run_config import (
    AugmentationConfig,
    DatasetConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    ScalerConfig,
    SchedulerConfig,
    SingleLossConfig,
)
from new_pipeline.datasets.base import StandardDataset
from new_pipeline.factories.model_factory import ModelFactory
from new_pipeline.factories.transform_factory import TransformFactory
from new_pipeline.orchestration.run_builder import RunBuilder
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = [
    {"type": "Resize", "height": 512, "width": 512},
    {"type": "HorizontalFlip", "p": 1.0},
    {
        "type": "OneOf",
        "transforms": [
            {"type": "RandomBrightnessContrast", "p": 1.0},
            {"type": "RandomGamma", "p": 0.0},
        ],
        "p": 0.0,
    },
    {"type": "Normalize", "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    {"type": "ToTensorV2"},
]

config = RunConfig(
    device=DEVICE,
    epochs=10,
    img_height=512,
    img_width=512,
    accumulation_steps=2,
    model=ModelConfig(type="dummy", kwargs={"num_classes": 3}),
    batch_size=8,
    loss=LossConfig(losses=[SingleLossConfig(type="CE", weight=1.0, kwargs={})]),
    optimizer=OptimizerConfig(type="ADAM", lr=0.001),
    scheduler=SchedulerConfig(type="CosineAnnealingLR", kwargs={"T_max": 10}),
    scaler=ScalerConfig(enabled=True),
    dataset=DatasetConfig(
        name="REFUGE",
        roi=False,
        local_root="datasets/REFUGE",
        drive_root="drive/datasets/REFUGE",
        augmentation=AugmentationConfig(
            train_transforms=transforms,
            val_transforms=transforms,
            multiplier=1,
        ),
        in_memory=True,
    ),
    root_exp_dir="runs/tests",
    visualization_samples="EVERY 2",
    visualization_epochs=[1, 3, 5, 7, 9],
)


def _dataset_root(dataset_config: DatasetConfig) -> str:
    if dataset_config.local_root and os.path.exists(dataset_config.local_root):
        return dataset_config.local_root
    if dataset_config.drive_root and os.path.exists(dataset_config.drive_root):
        return dataset_config.drive_root

    raise FileNotFoundError(
        f"Could not find dataset at {dataset_config.local_root!r} or {dataset_config.drive_root!r}."
    )


def minimal_run(run_config: RunConfig = config) -> None:
    """Plain train loop for rough per-epoch timing baselines."""
    device = run_config.device
    train_transforms, _ = TransformFactory.create(run_config.dataset.augmentation)
    train_ds = StandardDataset(
        root=os.path.join(_dataset_root(run_config.dataset), "train"),
        transforms=train_transforms,
        dataset_multiplier=run_config.dataset.augmentation.multiplier,
        n=run_config.dataset.num_of_samples,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=run_config.batch_size,
        shuffle=True,
        num_workers=run_config.num_workers,
    )

    model = ModelFactory.create(run_config.model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=run_config.optimizer.lr)

    print(f"Training {len(train_ds)} samples on {device} with {len(train_loader)} batches/epoch.")

    for epoch in range(1, run_config.epochs + 1):
        model.train()
        running_loss = 0.0

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        for images, masks, _ in train_loader:
            images = images.to(device).float()
            masks = masks.to(device).long()

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            #running_loss += loss.item()

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start
        avg_loss = running_loss / max(len(train_loader), 1)

        print(f"Epoch {epoch:03d}: {epoch_time:.2f}s | loss {avg_loss:.4f}")

def test_train_loader_time():
    run = RunBuilder().build(config)
    start = time.perf_counter()
    for i in range(1, 100):
        start = time.perf_counter()
        pbar = tqdm(enumerate(run.trainer.train_loader), total=len(run.trainer.train_loader), desc=f"Epoch {i}")
        for j, (images, masks, _) in pbar:
            image = images 
            mask = masks
        end = time.perf_counter()
        print(f"DataLoader iteration setup time: {end - start:.2f}s for 1 epoch")
    end = time.perf_counter()
    print(f"Total time for epochs: {end -start:.2f}s")

def run():
    run = RunBuilder().build(config)
    #print(config.visualization_samples)
    #print(run.report_generator.visualization_samples)
    #print(run.report_generator.visualization_epochs)
    run.execute()
    #run.report_generator.generate('runs/tests/REFUGE_MUL40_dummy_ADAM_CosineAnnealingLR_CE/events.out.tfevents.1780528664.63c9945ca3ea.2925.0')

if __name__ == "__main__":
    #minimal_run()
    run()
    #test_train_loader_time()