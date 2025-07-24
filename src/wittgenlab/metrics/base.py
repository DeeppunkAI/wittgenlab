"""
Base classes for evaluation metrics.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    All metrics should inherit from this class and implement the compute method.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the metric with configuration parameters.
        
        Args:
            **kwargs: Configuration parameters specific to the metric
        """
        self.config = kwargs
        self.name = self.__class__.__name__.replace('Metric', '').lower()
    
    @abstractmethod
    def compute(self, predictions: List[str], references: List[str]) -> Any:
        """
        Compute the metric score.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference/ground truth outputs
            
        Returns:
            Computed metric score(s)
        """
        pass
    
    def validate_inputs(self, predictions: List[str], references: List[str]):
        """
        Validate input format and content.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(predictions, list) or not isinstance(references, list):
            raise ValueError("Predictions and references must be lists")
        
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if len(predictions) == 0:
            raise ValueError("Input lists cannot be empty")
        
        # Check that all elements are strings
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            if not isinstance(pred, str):
                raise ValueError(f"Prediction at index {i} is not a string: {type(pred)}")
            if not isinstance(ref, str):
                raise ValueError(f"Reference at index {i} is not a string: {type(ref)}")
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess text before metric computation.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Default preprocessing: strip whitespace
        return text.strip()
    
    def postprocess(self, score: Any) -> Any:
        """
        Postprocess computed score.
        
        Args:
            score: Raw computed score
            
        Returns:
            Processed score
        """
        # Default: return as-is
        return score
    
    def get_info(self) -> Dict[str, str]:
        """
        Get information about this metric.
        
        Returns:
            Dictionary with metric information
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'description': self.__doc__ or "No description available",
            'config': str(self.config)
        }
    
    def __repr__(self) -> str:
        """String representation of the metric."""
        return f"{self.__class__.__name__}(config={self.config})"


class ReferenceBasedMetric(BaseMetric):
    """
    Base class for metrics that require reference texts.
    
    This is the most common type of metric that compares predictions
    against ground truth references.
    """
    
    def compute(self, predictions: List[str], references: List[str]) -> Any:
        """
        Compute reference-based metric.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            Computed metric score
        """
        self.validate_inputs(predictions, references)
        
        # Preprocess inputs
        predictions = [self.preprocess(pred) for pred in predictions]
        references = [self.preprocess(ref) for ref in references]
        
        # Compute score
        score = self._compute_score(predictions, references)
        
        # Postprocess score
        return self.postprocess(score)
    
    def compute_with_per_item(self, predictions: List[str], references: List[str]) -> tuple:
        """
        Compute reference-based metric with per-item scores.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            Tuple of (overall_score, per_item_scores)
        """
        self.validate_inputs(predictions, references)
        
        # Preprocess inputs
        predictions = [self.preprocess(pred) for pred in predictions]
        references = [self.preprocess(ref) for ref in references]
        
        # Compute overall score
        overall_score = self._compute_score(predictions, references)
        
        # Compute per-item scores
        per_item_scores = self._compute_per_item_scores(predictions, references)
        
        # Postprocess scores
        overall_score = self.postprocess(overall_score)
        per_item_scores = [self.postprocess_item(score) for score in per_item_scores]
        
        return overall_score, per_item_scores
    
    def _compute_per_item_scores(self, predictions: List[str], references: List[str]) -> List[Any]:
        """
        Compute per-item scores.
        
        Default implementation computes score for each prediction-reference pair individually.
        Subclasses can override this for more efficient computation.
        
        Args:
            predictions: List of predicted outputs
            references: List of reference outputs
            
        Returns:
            List of per-item scores
        """
        per_item_scores = []
        for pred, ref in zip(predictions, references):
            try:
                score = self._compute_score([pred], [ref])
                per_item_scores.append(score)
            except Exception as e:
                logger.warning(f"Error computing per-item score: {e}")
                per_item_scores.append(None)
        
        return per_item_scores
    
    def postprocess_item(self, score: Any) -> Any:
        """
        Postprocess individual item score.
        
        Default implementation uses the same postprocessing as overall score.
        Subclasses can override for item-specific postprocessing.
        
        Args:
            score: Raw computed score for a single item
            
        Returns:
            Processed score for the item
        """
        return self.postprocess(score)
    
    @abstractmethod
    def _compute_score(self, predictions: List[str], references: List[str]) -> Any:
        """
        Internal method to compute the actual score.
        
        Args:
            predictions: Preprocessed predictions
            references: Preprocessed references
            
        Returns:
            Raw computed score
        """
        pass


class ReferenceFreeMetric(BaseMetric):
    """
    Base class for metrics that don't require reference texts.
    
    These metrics evaluate properties of the predictions themselves,
    such as fluency, diversity, or coherence.
    """
    
    def compute(self, predictions: List[str], references: Optional[List[str]] = None) -> Any:
        """
        Compute reference-free metric.
        
        Args:
            predictions: List of predicted outputs
            references: Optional references (ignored for reference-free metrics)
            
        Returns:
            Computed metric score
        """
        if not isinstance(predictions, list):
            raise ValueError("Predictions must be a list")
        
        if len(predictions) == 0:
            raise ValueError("Predictions list cannot be empty")
        
        # Preprocess inputs
        predictions = [self.preprocess(pred) for pred in predictions]
        
        # Compute score
        score = self._compute_score(predictions)
        
        # Postprocess score
        return self.postprocess(score)
    
    @abstractmethod
    def _compute_score(self, predictions: List[str]) -> Any:
        """
        Internal method to compute the actual score.
        
        Args:
            predictions: Preprocessed predictions
            
        Returns:
            Raw computed score
        """
        pass 