"""
Benchmarks module for WittgenLab evaluation framework.

This module contains implementations of standardized benchmarks
organized by category: GLUE, knowledge, code, safety, and multilingual.
"""

from .registry import BenchmarksRegistry
from .base import BaseBenchmark

# Import benchmark categories
from . import glue
from . import knowledge
from . import code
from . import safety
from . import multilingual

__all__ = [
    "BenchmarksRegistry",
    "BaseBenchmark",
    "glue",
    "knowledge",
    "code",
    "safety",
    "multilingual"
] 