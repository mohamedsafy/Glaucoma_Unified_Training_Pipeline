from __future__ import annotations

from typing import Any, Optional, Union

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

        if config.visualization_samples is not None:
            visualization_samples = self.create_visualization_samples(
                config.visualization_samples,
                val_ds.ids,
            )

        exp_dir=os.path.join(config.root_exp_dir, config.short_desc)
        os.makedirs(exp_dir, exist_ok=True)
        report_generator = ReportGenerator(
            val_dataset=val_ds,
            config=config,
            writer=SummaryWriter(exp_dir),
            visualization_samples=visualization_samples,
            visualization_epochs=config.visualization_epochs,
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

    def create_visualization_samples(
        self,
        visualization_command: Union[str, list[str], list[int]],
        validation_sample_names: Optional[list[str]] = None,
    ) -> list[str]:
        '''
        Parses the visualization command to determine which samples to visualize. The command can be:
        - 'ALL': Visualize all samples in the validation dataset.
        - A comma-separated list of sample names: Visualize only the specified samples.
        - A comma-separated list of sample indices: Visualize the samples at the specified indices in the validation dataset.
        - 'EVERY N': Visualize every N-th sample in the validation dataset.
        - 'FIRST N': Visualize the first N samples in the validation dataset.
        - 'LAST N': Visualize the last N samples in the validation dataset.

        Returns a list of sample names to visualize from the validation dataset.
        '''
        sample_names = self._get_validation_sample_names(validation_sample_names)

        if isinstance(visualization_command, str):
            command = visualization_command.strip()
            if not command:
                return []

            command_parts = command.split()
            keyword = command_parts[0].upper()

            if keyword == "ALL" and len(command_parts) == 1:
                return list(sample_names)

            if keyword in {"EVERY", "FIRST", "LAST"}:
                if len(command_parts) != 2:
                    raise ValueError(
                        f"Visualization command {visualization_command!r} must be formatted as '{keyword} N'."
                    )
                n = self._parse_non_negative_int(command_parts[1], visualization_command)
                if keyword == "EVERY":
                    if n == 0:
                        raise ValueError("'EVERY N' requires N to be greater than 0.")
                    return list(sample_names[::n])
                if keyword == "FIRST":
                    return list(sample_names[:n])
                return list(sample_names[-n:] if n else [])

            tokens: list[str] = [token.strip() for token in command.split(",") if token.strip()]
            if not tokens:
                return []

            if all(self._is_int(token) for token in tokens):
                return self._sample_names_from_indices([int(token) for token in tokens], sample_names)

            return self._validate_sample_names(tokens, sample_names)

        if all(isinstance(item, int) for item in visualization_command):
            return self._sample_names_from_indices(visualization_command, sample_names)

        if all(isinstance(item, str) for item in visualization_command):
            tokens = [item.strip() for item in visualization_command if item.strip()]
            if all(self._is_int(token) for token in tokens):
                return self._sample_names_from_indices([int(token) for token in tokens], sample_names)
            return self._validate_sample_names(tokens, sample_names)

        raise TypeError(
            "visualization_command must be a string, a list of sample names, or a list of sample indices."
        )

    def _get_validation_sample_names(self, validation_sample_names: Optional[list[str]]) -> list[str]:
        if validation_sample_names is not None:
            return list(validation_sample_names)

        raise ValueError(
            "Validation sample names are required to resolve visualization sample commands."
        )

    def _parse_non_negative_int(self, value: str, command: str) -> int:
        if not self._is_int(value):
            raise ValueError(f"Visualization command {command!r} must end with an integer.")

        parsed_value = int(value)
        if parsed_value < 0:
            raise ValueError(f"Visualization command {command!r} requires a non-negative integer.")
        return parsed_value

    def _sample_names_from_indices(self, indices: list[int], sample_names: list[str]) -> list[str]:
        resolved_samples: list[str] = []
        for index in indices:
            if index < 0 or index >= len(sample_names):
                raise IndexError(
                    f"Visualization sample index {index} is out of range for "
                    f"{len(sample_names)} validation samples."
                )
            resolved_samples.append(sample_names[index])

        return resolved_samples

    def _validate_sample_names(self, requested_names: list[str], sample_names: list[str]) -> list[str]:
        available_names = set(sample_names)
        missing_names = [name for name in requested_names if name not in available_names]
        if missing_names:
            raise ValueError(
                "Visualization samples not found in validation dataset: "
                + ", ".join(missing_names)
            )

        return requested_names

    def _is_int(self, value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False



