from __future__ import annotations

from typing import Any, Optional

from new_pipeline.config import ScalerConfig


class ScalerFactory:
    @classmethod
    def create(cls, config: ScalerConfig) -> Optional[Any]:
        if not config.enabled:
            return None

        try:
            from torch.cuda.amp import GradScaler
        except ImportError as exc:
            raise RuntimeError("AMP scaling requires PyTorch to be installed.") from exc

        return GradScaler(**config.kwargs)
