from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from new_pipeline.config import RunConfig
from new_pipeline.reporting import ReportGenerator
from new_pipeline.runtime.trainer import Trainer


@dataclass
class Run:
    config: RunConfig
    trainer: Trainer
    report_generator: ReportGenerator

    def execute(self) -> None:
        ex=None
        try:
            self.trainer.train()
            self.report_generator.generate()
        except Exception as ex:
            print(ex)

        #self.report_generator.save_metadata(ex)
        
        self.clean()


    def save_experiment_info(self) -> None:
        raise NotImplementedError("Experiment metadata persistence is not implemented in the skeleton.")

    def clean(self) -> None:
        self.trainer.clean()
        self.report_generator.clean()

