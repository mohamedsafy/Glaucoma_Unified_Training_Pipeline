from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar


TConfig = TypeVar("TConfig")


class FactoryNotRegisteredError(ValueError):
    """Raised when a skeleton factory has no registered implementation."""


class RegistryFactory(Generic[TConfig]):
    registry: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, key: str, builder: Callable[..., Any]) -> None:
        cls.registry[key] = builder

    @classmethod
    def resolve(cls, key: str) -> Callable[..., Any]:
        try:
            return cls.registry[key]
        except KeyError as exc:
            available = ", ".join(sorted(cls.registry)) or "none"
            message = f"No builder registered for {key!r}. Available builders: {available}."
            raise FactoryNotRegisteredError(message) from exc
