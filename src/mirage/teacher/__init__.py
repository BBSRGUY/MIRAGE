from .base import TeacherAdapter
from .ltx import LTXTeacherAdapter
from .registry import get_teacher_adapter, register_teacher, registered_teachers

if "ltx25" not in registered_teachers():
    register_teacher("ltx25", LTXTeacherAdapter)

__all__ = [
    "LTXTeacherAdapter",
    "TeacherAdapter",
    "get_teacher_adapter",
    "register_teacher",
    "registered_teachers",
]
