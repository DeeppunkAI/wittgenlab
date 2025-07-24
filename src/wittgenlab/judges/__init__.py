"""
LLM-as-a-Judge implementations for WittgenLab.

This module provides various judge implementations for evaluating
text quality using language models.
"""

from .base import BaseJudge, Judge, JudgeResult, ConsensusJudge
from .scoring.llm_judge import LLMJudge, MultiModelJudge, create_judge

__all__ = [
    "BaseJudge",
    "Judge", 
    "JudgeResult",
    "ConsensusJudge",
    "LLMJudge",
    "MultiModelJudge",
    "create_judge"
] 