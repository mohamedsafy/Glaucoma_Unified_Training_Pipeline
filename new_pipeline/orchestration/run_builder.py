from __future__ import annotations

from new_pipeline.orchestration.dataset_builder import DatasetBuilder

from new_pipeline.config import RunConfig
from new_pipeline.factories import (
    DatasetFactory,
    LossFactory,
    ModelFactory,
    OptimizerFactory,
    ScalerFactory,
    SchedulerFactory,
)
from new_pipeline.orchestration.run import Run
from new_pipeline.reporting import ReportGenerator
from new_pipeline.runtime.data_module import DataModule
from new_pipeline.runtime.trainer import Trainer

from torch.utils.tensorboard import SummaryWriter
import os


class RunBuilder:
    def build(self, config: RunConfig) -> Run:
        model = ModelFactory.create(config.model)
        # Ensure model and loss are on the configured device so AMP/autocast
        # and CUDA tensors don't end up mismatched (Half vs Float) at runtime.
        model = model.to(config.device)
        criterion = LossFactory.create(config.loss)
        criterion = criterion.to(config.device) if hasattr(criterion, "to") else criterion
        optimizer = OptimizerFactory.create(config.optimizer, model.parameters())
        scheduler = SchedulerFactory.create(config.scheduler, optimizer)
        scaler = ScalerFactory.create(config.scaler)
        train_ds, val_ds, test_ds = DatasetBuilder().build(config.dataset)

        data_module = DataModule(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        if config.visualization_samples == "ALL":
            config.visualization_samples = val_ds.ids

        exp_dir=os.path.join(config.root_exp_dir, config.short_desc)
        os.makedirs(exp_dir, exist_ok=True)
        report_generator = ReportGenerator(
            val_dataset=val_ds,
            config=config,
            writer=SummaryWriter(exp_dir),
            visualization_samples=config.visualization_samples,
            exp_title=config.short_desc,
            exp_dir=exp_dir,
        )


        trainer = Trainer(
            device=config.device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_module=data_module,
            epochs=config.epochs,
            accumulation_steps=config.accumulation_steps,
            visualization_samples=config.visualization_samples,
            report_generator=report_generator,
        )


        return Run(config=config, trainer=trainer, report_generator=report_generator)
