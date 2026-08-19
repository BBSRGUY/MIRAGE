"""MIRAGE: Motion-Invariant Residual Adaptive Generative Engine."""

from .config import MirageConfig
from .model import MirageGenerator

__all__ = ["MirageConfig", "MirageGenerator"]
__version__ = "0.1.0"
