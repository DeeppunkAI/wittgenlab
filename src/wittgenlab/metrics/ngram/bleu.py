"""
BLEU (Bilingual Evaluation Understudy) metric implementation.
"""

from typing import List, Dict, Any
import logging

from ..base import ReferenceBasedMetric

logger = logging.getLogger(__name__)


class BLEUMetric(ReferenceBasedMetric):
    """
    BLEU metric for evaluating text generation quality.
    
    BLEU measures the overlap of n-grams between the predicted text
    and reference text, with a brevity penalty for short predictions.
    """
    
    def __init__(self, max_order: int = 4, smooth: bool = False, **kwargs):
        """
        Initialize BLEU metric.
        
        Args:
            max_order: Maximum n-gram order to consider (default: 4)
            smooth: Whether to apply smoothing for better estimates
            **kwargs: Additional configuration parameters
        """
        super().__init__(max_order=max_order, smooth=smooth, **kwargs)
        self.max_order = max_order
        self.smooth = smooth
    
    def _compute_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute BLEU score.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU scores
        """
        try:
            # Try to use sacrebleu if available
            import sacrebleu
            
            # Compute corpus-level BLEU
            bleu = sacrebleu.corpus_bleu(predictions, [references])
            
            result = {
                'bleu': bleu.score / 100.0,  # Convert to 0-1 scale
                'bleu_1': bleu.precisions[0] / 100.0 if len(bleu.precisions) > 0 else 0.0,
                'bleu_2': bleu.precisions[1] / 100.0 if len(bleu.precisions) > 1 else 0.0,
                'bleu_3': bleu.precisions[2] / 100.0 if len(bleu.precisions) > 2 else 0.0,
                'bleu_4': bleu.precisions[3] / 100.0 if len(bleu.precisions) > 3 else 0.0,
                'brevity_penalty': bleu.bp,
            }
            
        except ImportError:
            logger.warning("sacrebleu not available, using simplified BLEU implementation")
            result = self._simple_bleu(predictions, references)
        
        return result
    
    def _simple_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Simplified BLEU implementation without external dependencies.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU score
        """
        import collections
        import math
        
        def _get_ngrams(text: str, n: int) -> collections.Counter:
            """Extract n-grams from text."""
            words = text.lower().split()
            return collections.Counter(
                tuple(words[i:i+n]) for i in range(len(words) - n + 1)
            ) if len(words) >= n else collections.Counter()
        
        # Compute n-gram precisions
        precisions = []
        
        for n in range(1, self.max_order + 1):
            pred_ngrams = collections.Counter()
            ref_ngrams = collections.Counter()
            
            for pred, ref in zip(predictions, references):
                pred_ngrams.update(_get_ngrams(pred, n))
                ref_ngrams.update(_get_ngrams(ref, n))
            
            # Compute precision for this n-gram order
            if sum(pred_ngrams.values()) == 0:
                precision = 0.0
            else:
                matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                            for ngram in pred_ngrams)
                precision = matches / sum(pred_ngrams.values())
            
            precisions.append(precision)
        
        # Compute brevity penalty
        pred_len = sum(len(pred.split()) for pred in predictions)
        ref_len = sum(len(ref.split()) for ref in references)
        
        if pred_len == 0:
            bp = 0.0
        elif pred_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / pred_len)
        
        # Compute geometric mean of precisions
        if any(p == 0 for p in precisions):
            bleu_score = 0.0
        else:
            log_precisions = [math.log(p) for p in precisions]
            bleu_score = bp * math.exp(sum(log_precisions) / len(log_precisions))
        
        return {
            'bleu': bleu_score,
            'bleu_1': precisions[0] if len(precisions) > 0 else 0.0,
            'bleu_2': precisions[1] if len(precisions) > 1 else 0.0,
            'bleu_3': precisions[2] if len(precisions) > 2 else 0.0,
            'bleu_4': precisions[3] if len(precisions) > 3 else 0.0,
            'brevity_penalty': bp,
        }
    
    def postprocess(self, score: Dict[str, float]) -> float:
        """
        Postprocess BLEU score.
        
        Args:
            score: Raw BLEU score dictionary
            
        Returns:
            Main BLEU score (float)
        """
        # Return main BLEU score by default
        return score.get('bleu', 0.0) 