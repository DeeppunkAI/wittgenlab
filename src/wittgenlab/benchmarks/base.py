"""
Base classes for evaluation benchmarks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """
    Abstract base class for all evaluation benchmarks.
    
    All benchmarks should inherit from this class and implement the evaluate method.
    """
    
    def __init__(self, few_shot: int = 0, batch_size: int = 1, **kwargs):
        """
        Initialize the benchmark with configuration parameters.
        
        Args:
            few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            **kwargs: Additional configuration parameters
        """
        self.few_shot = few_shot
        self.batch_size = batch_size
        self.config = kwargs
        self.name = self.__class__.__name__.replace('Benchmark', '').lower()
    
    @abstractmethod
    def evaluate(self, model: Any) -> Dict[str, Any]:
        """
        Evaluate a model on this benchmark.
        
        Args:
            model: The model to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        pass
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load the benchmark data.
        
        Returns:
            List of data samples
        """
        pass
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess the loaded data.
        
        Args:
            data: Raw data samples
            
        Returns:
            Preprocessed data samples
        """
        # Default: return as-is
        return data
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a data sample into a prompt for the model.
        
        Args:
            sample: Data sample
            
        Returns:
            Formatted prompt string
        """
        # Default implementation - should be overridden
        return str(sample)
    
    def parse_response(self, response: str, sample: Dict[str, Any]) -> Any:
        """
        Parse model response to extract the answer.
        
        Args:
            response: Raw model response
            sample: Original data sample
            
        Returns:
            Parsed answer
        """
        # Default: return response as-is
        return response.strip()
    
    def compute_score(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        """
        Compute benchmark score from predictions and references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            
        Returns:
            Dictionary with computed scores
        """
        # Default: accuracy for classification tasks
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        accuracy = correct / len(predictions)
        
        return {
            'accuracy': accuracy,
            'total_samples': len(predictions),
            'correct_samples': correct
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this benchmark.
        
        Returns:
            Dictionary with benchmark information
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'description': self.__doc__ or "No description available",
            'few_shot': self.few_shot,
            'batch_size': self.batch_size,
            'config': self.config
        }
    
    def __repr__(self) -> str:
        """String representation of the benchmark."""
        return f"{self.__class__.__name__}(few_shot={self.few_shot}, batch_size={self.batch_size})"


class MultipleChoiceBenchmark(BaseBenchmark):
    """
    Base class for multiple choice benchmarks.
    """
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a multiple choice sample into a prompt.
        
        Args:
            sample: Data sample with 'question' and 'choices' keys
            
        Returns:
            Formatted prompt string
        """
        question = sample.get('question', '')
        choices = sample.get('choices', [])
        
        prompt = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65 + i)}. {choice}\n"
        
        prompt += "\nAnswer:"
        return prompt
    
    def parse_response(self, response: str, sample: Dict[str, Any]) -> str:
        """
        Parse multiple choice response to extract the selected option.
        
        Args:
            response: Raw model response
            sample: Original data sample
            
        Returns:
            Selected option (A, B, C, D, etc.)
        """
        response = response.strip().upper()
        
        # Extract the first letter that looks like a choice
        for char in response:
            if char in 'ABCDEFGHIJ':
                return char
        
        # If no clear choice found, return the first character
        return response[0] if response else 'A'


class GenerationBenchmark(BaseBenchmark):
    """
    Base class for text generation benchmarks.
    """
    
    def format_prompt(self, sample: Dict[str, Any]) -> str:
        """
        Format a generation sample into a prompt.
        
        Args:
            sample: Data sample with 'prompt' or 'input' key
            
        Returns:
            Formatted prompt string
        """
        return sample.get('prompt', sample.get('input', ''))
    
    def compute_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute generation benchmark score using text metrics.
        
        Args:
            predictions: Generated texts
            references: Reference texts
            
        Returns:
            Dictionary with computed scores
        """
        # Use BLEU and ROUGE as default metrics for generation
        from ..metrics.ngram.bleu import BLEUMetric
        from ..metrics.ngram.rouge import ROUGEMetric
        
        bleu_metric = BLEUMetric()
        rouge_metric = ROUGEMetric()
        
        bleu_score = bleu_metric.compute(predictions, references)
        rouge_scores = rouge_metric._compute_score(predictions, references)
        
        result = {
            'bleu': bleu_score,
            'total_samples': len(predictions)
        }
        result.update(rouge_scores)
        
        return result 