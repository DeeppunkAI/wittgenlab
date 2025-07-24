"""
N-gram based evaluation metrics.

This module contains implementations of traditional n-gram based metrics
like BLEU, ROUGE, METEOR, and CIDEr.
"""

from .bleu import BLEUMetric
from .rouge import ROUGEMetric

__all__ = ["BLEUMetric", "ROUGEMetric"] 