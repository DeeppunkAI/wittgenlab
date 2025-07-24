"""
WittgenLab - A comprehensive evaluation framework for AI models.

This package provides tools for evaluating machine learning models across
various metrics, benchmarks, and evaluation methodologies.
"""

from .core.evaluator import EvalHub
from .core.config import EvalConfig
from .core.results import EvalResults, BenchmarkResults

__version__ = "0.1.0"
__author__ = "Robert Gomez"
__email__ = "robertgomez.datascience@gmail.com"

__all__ = [
    "EvalHub",
    "EvalConfig", 
    "EvalResults",
    "BenchmarkResults",
] 