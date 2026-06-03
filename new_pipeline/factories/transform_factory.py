from __future__ import annotations

from typing import Any

from new_pipeline.config import AugmentationConfig


class TransformFactory:
    """Build Albumentations pipelines from transform objects or dictionaries.

    Example item:
        {"type": "Resize", "height": 512, "width": 512}

    Nested transforms:
        {"type": "OneOf", "transforms": [{"type": "ElasticTransform"}], "p": 0.7}
    """

    @classmethod
    def create(cls, config: AugmentationConfig) -> tuple[Any, Any]:
        return (
            cls._compose(config.train_transforms),
            cls._compose(config.val_transforms),
        )

    @classmethod
    def _compose(cls, transforms: Any) -> Any:
        if transforms is None:
            return None

        if not isinstance(transforms, list):
            return transforms

        if not transforms:
            return None

        albumentations = cls._albumentations()
        return albumentations.Compose([cls._build_transform(transform) for transform in transforms])

    @classmethod
    def _build_transform(cls, spec: Any) -> Any:
        if not isinstance(spec, dict):
            return spec

        transform_name = spec.get("type") or spec.get("name") or spec.get("transform")
        if transform_name is None:
            raise ValueError(f"Transform dictionary must include a type/name/transform key: {spec}")

        kwargs = dict(spec.get("kwargs", {}))
        for key, value in spec.items():
            if key not in {"type", "name", "transform", "kwargs"}:
                kwargs[key] = value

        nested_specs = kwargs.pop("transforms", None)
        if nested_specs is not None:
            kwargs["transforms"] = [cls._build_transform(nested_spec) for nested_spec in nested_specs]

        kwargs = {key: cls._normalize_value(value) for key, value in kwargs.items()}

        if transform_name == "ToTensorV2":
            transform_cls = cls._to_tensor_v2()
        else:
            transform_cls = getattr(cls._albumentations(), transform_name)

        return transform_cls(**kwargs)

    @staticmethod
    def _albumentations() -> Any:
        try:
            import albumentations as A
        except ImportError as exc:
            raise RuntimeError("Albumentations is required to build dictionary-based transforms.") from exc

        return A

    @staticmethod
    def _to_tensor_v2() -> Any:
        try:
            from albumentations.pytorch import ToTensorV2
        except ImportError as exc:
            raise RuntimeError("albumentations.pytorch.ToTensorV2 is required for ToTensorV2 transforms.") from exc

        return ToTensorV2

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        if isinstance(value, str) and value.startswith("cv2."):
            return cls._resolve_cv2_constant(value)

        if isinstance(value, list):
            return tuple(cls._normalize_value(item) for item in value)

        if isinstance(value, dict):
            return {key: cls._normalize_value(item) for key, item in value.items()}

        return value

    @staticmethod
    def _resolve_cv2_constant(value: str) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(f"OpenCV is required to resolve {value!r}.") from exc

        _, constant_name = value.split(".", 1)
        return getattr(cv2, constant_name)
