"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric implementation.
"""

from typing import List, Dict, Any
import logging

from ..base import ReferenceBasedMetric

logger = logging.getLogger(__name__)


class ROUGEMetric(ReferenceBasedMetric):
    """
    ROUGE metric for evaluating text summarization quality.
    
    ROUGE measures the overlap of n-grams and word sequences between
    the predicted text and reference text, focusing on recall.
    """
    
    def __init__(self, rouge_types: List[str] = None, **kwargs):
        """
        Initialize ROUGE metric.
        
        Args:
            rouge_types: List of ROUGE types to compute (e.g., ['rouge1', 'rouge2', 'rougeL'])
            **kwargs: Additional configuration parameters
        """
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        
        super().__init__(rouge_types=rouge_types, **kwargs)
        self.rouge_types = rouge_types
    
    def _compute_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        try:
            # Try to use rouge-score package if available
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
            
            all_scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} 
                         for rouge_type in self.rouge_types}
            
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                
                for rouge_type in self.rouge_types:
                    if rouge_type in scores:
                        all_scores[rouge_type]['precision'].append(scores[rouge_type].precision)
                        all_scores[rouge_type]['recall'].append(scores[rouge_type].recall)
                        all_scores[rouge_type]['fmeasure'].append(scores[rouge_type].fmeasure)
            
            # Compute averages
            result = {}
            for rouge_type in self.rouge_types:
                if all_scores[rouge_type]['fmeasure']:
                    result[f'{rouge_type}_precision'] = sum(all_scores[rouge_type]['precision']) / len(all_scores[rouge_type]['precision'])
                    result[f'{rouge_type}_recall'] = sum(all_scores[rouge_type]['recall']) / len(all_scores[rouge_type]['recall'])
                    result[f'{rouge_type}_f1'] = sum(all_scores[rouge_type]['fmeasure']) / len(all_scores[rouge_type]['fmeasure'])
                    
                    # Main score is F1
                    result[rouge_type] = result[f'{rouge_type}_f1']
            
        except ImportError:
            logger.warning("rouge-score package not available, using simplified ROUGE implementation")
            result = self._simple_rouge(predictions, references)
        
        return result
    
    def _compute_per_item_scores(self, predictions: List[str], references: List[str]) -> List[Any]:
        """
        Compute per-item ROUGE scores efficiently.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            List of per-item ROUGE scores
        """
        try:
            # Try to use rouge-score package if available
            from rouge_score import rouge_scorer
            
            scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
            
            per_item_scores = []
            for pred, ref in zip(predictions, references):
                scores = scorer.score(ref, pred)
                
                item_result = {}
                for rouge_type in self.rouge_types:
                    if rouge_type in scores:
                        item_result[f'{rouge_type}_precision'] = scores[rouge_type].precision
                        item_result[f'{rouge_type}_recall'] = scores[rouge_type].recall
                        item_result[f'{rouge_type}_f1'] = scores[rouge_type].fmeasure
                        
                        # Main score is F1
                        item_result[rouge_type] = scores[rouge_type].fmeasure
                
                per_item_scores.append(item_result)
            
        except ImportError:
            logger.warning("rouge-score package not available, using simplified ROUGE implementation for per-item scores")
            per_item_scores = []
            for pred, ref in zip(predictions, references):
                score = self._simple_rouge([pred], [ref])
                per_item_scores.append(score)
        
        return per_item_scores
    
    def _simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Simplified ROUGE implementation without external dependencies.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE scores
        """
        import collections
        
        def _get_ngrams(text: str, n: int) -> collections.Counter:
            """Extract n-grams from text."""
            words = text.lower().split()
            return collections.Counter(
                tuple(words[i:i+n]) for i in range(len(words) - n + 1)
            ) if len(words) >= n else collections.Counter()
        
        def _lcs_length(x: List[str], y: List[str]) -> int:
            """Compute length of longest common subsequence."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        result = {}
        
        # ROUGE-1 and ROUGE-2
        for n in [1, 2]:
            if f'rouge{n}' in self.rouge_types:
                precision_scores = []
                recall_scores = []
                f1_scores = []
                
                for pred, ref in zip(predictions, references):
                    pred_ngrams = _get_ngrams(pred, n)
                    ref_ngrams = _get_ngrams(ref, n)
                    
                    if sum(pred_ngrams.values()) == 0:
                        precision = 0.0
                    else:
                        matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                                    for ngram in pred_ngrams)
                        precision = matches / sum(pred_ngrams.values())
                    
                    if sum(ref_ngrams.values()) == 0:
                        recall = 0.0
                    else:
                        matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                                    for ngram in ref_ngrams)
                        recall = matches / sum(ref_ngrams.values())
                    
                    if precision + recall == 0:
                        f1 = 0.0
                    else:
                        f1 = 2 * precision * recall / (precision + recall)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                
                result[f'rouge{n}_precision'] = sum(precision_scores) / len(precision_scores)
                result[f'rouge{n}_recall'] = sum(recall_scores) / len(recall_scores)
                result[f'rouge{n}_f1'] = sum(f1_scores) / len(f1_scores)
                result[f'rouge{n}'] = result[f'rouge{n}_f1']
        
        # ROUGE-L
        if 'rougeL' in self.rouge_types:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_words = pred.lower().split()
                ref_words = ref.lower().split()
                
                lcs_len = _lcs_length(pred_words, ref_words)
                
                if len(pred_words) == 0:
                    precision = 0.0
                else:
                    precision = lcs_len / len(pred_words)
                
                if len(ref_words) == 0:
                    recall = 0.0
                else:
                    recall = lcs_len / len(ref_words)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            
            result['rougeL_precision'] = sum(precision_scores) / len(precision_scores)
            result['rougeL_recall'] = sum(recall_scores) / len(recall_scores)
            result['rougeL_f1'] = sum(f1_scores) / len(f1_scores)
            result['rougeL'] = result['rougeL_f1']
        
        return result
    
    def postprocess_item(self, score: Dict[str, float]) -> float:
        """
        Postprocess individual ROUGE score.
        
        Args:
            score: Raw ROUGE score dictionary for a single item
            
        Returns:
            Main ROUGE score (rouge1 F1 by default) for the item
        """
        # Return ROUGE-1 F1 as the main score for individual items
        return score.get('rouge1', score.get('rouge1_f1', 0.0))

    def postprocess(self, score: Dict[str, float]) -> float:
        """
        Postprocess ROUGE score.
        
        Args:
            score: Raw ROUGE score dictionary
            
        Returns:
            Main ROUGE score (rouge1 F1 by default)
        """
        # Return ROUGE-1 F1 as the main score
        return score.get('rouge1', score.get('rouge1_f1', 0.0)) 