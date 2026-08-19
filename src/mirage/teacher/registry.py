from __future__ import annotations

from collections.abc import Callable

from ..m2_config import TeacherConfig
from .base import TeacherAdapter

AdapterFactory = Callable[[TeacherConfig], TeacherAdapter]
_REGISTRY: dict[str, AdapterFactory] = {}


def register_teacher(name: str, factory: AdapterFactory) -> None:
    if not name or name in _REGISTRY:
        raise ValueError(f"teacher adapter already registered or invalid: {name}")
    _REGISTRY[name] = factory


def get_teacher_adapter(name: str, config: TeacherConfig) -> TeacherAdapter:
    try:
        return _REGISTRY[name](config)
    except KeyError as error:
        raise ValueError(f"unsupported teacher {name!r}; available: {sorted(_REGISTRY)}") from error


def registered_teachers() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))
