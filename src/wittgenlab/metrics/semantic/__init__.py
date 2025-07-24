"""
Semantic similarity evaluation metrics.

This module contains implementations of semantic similarity metrics
like BERTScore, MoverScore, and BLEURT.
"""

from .bertscore import BERTScoreMetric

# Placeholder for other semantic metrics
# from .moverscore import MoverScoreMetric
# from .bleurt import BLEURTMetric

__all__ = ["BERTScoreMetric"] 