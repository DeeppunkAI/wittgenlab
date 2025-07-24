"""
Base classes for LLM-as-a-Judge evaluation.

This module provides the foundation for using language models to evaluate
text quality, accuracy, and other criteria.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class Judge(BaseModel):
    """
    Pydantic model for structured judge output.
    
    This model ensures consistent output format from LLM judges.
    """
    score: int = Field(description="Score between 0 and 5 used to qualify the response", ge=0, le=5)
    justification: str = Field(description="Justification of the score explaining the reasoning")


class JudgeResult(BaseModel):
    """
    Container for judge evaluation results.
    """
    score: int
    justification: str
    criterion: str
    model_name: str
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseJudge(ABC):
    """
    Abstract base class for all LLM judges.
    
    All judge implementations should inherit from this class.
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Initialize the judge.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            **kwargs: Additional model parameters
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
        
        # Initialize the LLM
        self._llm = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the language model."""
        pass
    
    @abstractmethod
    def evaluate(
        self,
        prediction: str,
        reference: Optional[str] = None,
        criterion: str = "quality",
        context: Optional[str] = None,
        **kwargs
    ) -> JudgeResult:
        """
        Evaluate a prediction using the judge.
        
        Args:
            prediction: The text to evaluate
            reference: Optional reference text
            criterion: Evaluation criterion
            context: Optional context for evaluation
            **kwargs: Additional parameters
            
        Returns:
            JudgeResult with score and justification
        """
        pass


class ConsensusJudge:
    """
    Class for combining multiple judge results using consensus methods.
    """
    
    def __init__(self, consensus_method: Literal["majority_vote", "average", "weighted_average"] = "majority_vote"):
        """
        Initialize consensus judge.
        
        Args:
            consensus_method: Method for combining judge results
        """
        self.consensus_method = consensus_method
    
    def combine_results(
        self,
        results: List[JudgeResult],
        weights: Optional[List[float]] = None
    ) -> JudgeResult:
        """
        Combine multiple judge results using consensus method.
        
        Args:
            results: List of individual judge results
            weights: Optional weights for weighted average
            
        Returns:
            Combined JudgeResult
        """
        if not results:
            raise ValueError("No results to combine")
        
        if self.consensus_method == "majority_vote":
            return self._majority_vote(results)
        elif self.consensus_method == "average":
            return self._average_score(results)
        elif self.consensus_method == "weighted_average":
            return self._weighted_average(results, weights or [1.0] * len(results))
        else:
            raise ValueError(f"Unknown consensus method: {self.consensus_method}")
    
    def _majority_vote(self, results: List[JudgeResult]) -> JudgeResult:
        """Combine results using majority vote."""
        from collections import Counter
        
        scores = [result.score for result in results]
        score_counts = Counter(scores)
        consensus_score = score_counts.most_common(1)[0][0]
        
        # Find a result with the consensus score for justification
        consensus_result = next(r for r in results if r.score == consensus_score)
        
        return JudgeResult(
            score=consensus_score,
            justification=f"Consensus score {consensus_score} from {len(results)} judges: " + 
                         consensus_result.justification,
            criterion=results[0].criterion,
            model_name="consensus",
            metadata={
                "method": "majority_vote",
                "individual_scores": scores,
                "individual_models": [r.model_name for r in results]
            }
        )
    
    def _average_score(self, results: List[JudgeResult]) -> JudgeResult:
        """Combine results using average score."""
        scores = [result.score for result in results]
        avg_score = round(sum(scores) / len(scores))
        
        # Find the result closest to average for justification
        closest_result = min(results, key=lambda r: abs(r.score - avg_score))
        
        return JudgeResult(
            score=avg_score,
            justification=f"Average score {avg_score:.1f} from {len(results)} judges: " + 
                         closest_result.justification,
            criterion=results[0].criterion,
            model_name="consensus",
            metadata={
                "method": "average",
                "individual_scores": scores,
                "exact_average": sum(scores) / len(scores),
                "individual_models": [r.model_name for r in results]
            }
        )
    
    def _weighted_average(self, results: List[JudgeResult], weights: List[float]) -> JudgeResult:
        """Combine results using weighted average."""
        if len(results) != len(weights):
            raise ValueError("Number of results must match number of weights")
        
        weighted_sum = sum(r.score * w for r, w in zip(results, weights))
        weight_sum = sum(weights)
        weighted_avg = round(weighted_sum / weight_sum)
        
        # Find the result closest to weighted average for justification
        closest_result = min(results, key=lambda r: abs(r.score - weighted_avg))
        
        return JudgeResult(
            score=weighted_avg,
            justification=f"Weighted average score {weighted_avg:.1f} from {len(results)} judges: " + 
                         closest_result.justification,
            criterion=results[0].criterion,
            model_name="consensus",
            metadata={
                "method": "weighted_average",
                "individual_scores": [r.score for r in results],
                "weights": weights,
                "exact_weighted_average": weighted_sum / weight_sum,
                "individual_models": [r.model_name for r in results]
            }
        ) 