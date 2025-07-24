"""
BERTScore metric implementation.

BERTScore leverages pre-trained contextualized embeddings from BERT 
to evaluate text similarity based on semantic content.
"""

from typing import List, Any, Dict, Optional, Union
import logging

from ..base import ReferenceBasedMetric

logger = logging.getLogger(__name__)


class BERTScoreMetric(ReferenceBasedMetric):
    """
    BERTScore metric for semantic similarity evaluation.
    
    This implementation uses the bert_score library directly for more reliable results.
    """
    
    def __init__(
        self,
        lang: str = "es",
        model_type: Optional[str] = None,
        num_layers: Optional[int] = None,
        verbose: bool = False,
        idf: bool = False,
        device: Optional[str] = None,
        batch_size: int = 64,
        nthreads: int = 4,
        all_layers: bool = False,
        rescale_with_baseline: bool = False,
        **kwargs
    ):
        """
        Initialize BERTScore metric.
        
        Args:
            lang: Language code (e.g., 'es', 'en', 'fr')
            model_type: Specific BERT model to use (if None, uses default for language)
            num_layers: Number of layers to use (if None, uses default)
            verbose: Whether to print verbose information
            idf: Whether to use inverse document frequency re-weighting
            device: Device to run on ('cpu', 'cuda', etc.)
            batch_size: Batch size for processing
            nthreads: Number of threads for processing
            all_layers: Whether to use all layers for scoring
            rescale_with_baseline: Whether to rescale with baseline
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        self.lang = lang
        self.model_type = model_type
        self.num_layers = num_layers
        self.verbose = verbose
        self.idf = idf
        self.device = device
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers
        self.rescale_with_baseline = rescale_with_baseline
        
        # Verify that bert_score is available
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify that bert_score is installed."""
        try:
            import bert_score
            logger.info("BERTScore library verified")
        except ImportError:
            raise ImportError(
                "bert-score is required for BERTScore metric. "
                "Install it with: pip install bert-score"
            )
    
    def _compute_score(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, Any]:
        """
        Compute BERTScore between predictions and references.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        try:
            # Import bert_score
            from bert_score import score
            
            # Validate inputs
            if not predictions or not references:
                raise ValueError("Predictions and references cannot be empty")
            
            if len(predictions) != len(references):
                raise ValueError("Predictions and references must have the same length")
            
            # Prepare arguments for bert_score.score
            score_args = {
                'lang': self.lang,
                'verbose': self.verbose,
                'idf': self.idf,
                'batch_size': self.batch_size,
                'nthreads': self.nthreads,
                'all_layers': self.all_layers,
                'rescale_with_baseline': self.rescale_with_baseline
            }
            
            # Add optional arguments if they are not None
            if self.model_type is not None:
                score_args['model_type'] = self.model_type
            
            if self.num_layers is not None:
                score_args['num_layers'] = self.num_layers
                
            if self.device is not None:
                score_args['device'] = self.device
            
            # Compute BERTScore
            logger.info(f"Computing BERTScore for {len(predictions)} samples with lang='{self.lang}'")
            
            P, R, F1 = score(
                cands=predictions,
                refs=references,
                **score_args
            )
            
            # Convert tensors to Python floats
            precision_mean = float(P.mean().item()) if hasattr(P, 'mean') else float(P)
            recall_mean = float(R.mean().item()) if hasattr(R, 'mean') else float(R)
            f1_mean = float(F1.mean().item()) if hasattr(F1, 'mean') else float(F1)
            
            # Create result dictionary
            result = {
                "precision": precision_mean,
                "recall": recall_mean,
                "f1": f1_mean
            }
            
            # Add individual scores if requested
            if self.config.get("return_individual_scores", False):
                result["precision_scores"] = [float(p.item()) for p in P]
                result["recall_scores"] = [float(r.item()) for r in R]
                result["f1_scores"] = [float(f.item()) for f in F1]
            
            logger.info(f"BERTScore computed successfully: F1={f1_mean:.4f}")
            return result
            
        except ImportError as e:
            error_msg = "bert-score library not found. Install with: pip install bert-score"
            logger.error(error_msg)
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"Error computing BERTScore: {str(e)}"
            logger.error(error_msg)
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "error": error_msg
            }
    
    def _compute_per_item_scores(self, predictions: List[str], references: List[str]) -> List[Any]:
        """
        Compute per-item BERTScore scores efficiently.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            List of per-item BERTScore scores (each containing precision, recall, f1)
        """
        try:
            # Import bert_score
            from bert_score import score
            
            # Prepare arguments for bert_score.score
            score_args = {
                'lang': self.lang,
                'verbose': False,  # Disable verbose for per-item computation
                'idf': self.idf,
                'batch_size': self.batch_size,
                'nthreads': self.nthreads,
                'all_layers': self.all_layers,
                'rescale_with_baseline': self.rescale_with_baseline
            }
            
            # Add optional arguments if they are not None
            if self.model_type is not None:
                score_args['model_type'] = self.model_type
            
            if self.num_layers is not None:
                score_args['num_layers'] = self.num_layers
                
            if self.device is not None:
                score_args['device'] = self.device
            
            # Compute BERTScore for all pairs
            logger.info(f"Computing per-item BERTScore for {len(predictions)} samples")
            
            P, R, F1 = score(
                cands=predictions,
                refs=references,
                **score_args
            )
            
            # Convert to individual score dictionaries
            per_item_scores = []
            for i, (p, r, f1) in enumerate(zip(P, R, F1)):
                item_score = {
                    "precision": float(p.item()) if hasattr(p, 'item') else float(p),
                    "recall": float(r.item()) if hasattr(r, 'item') else float(r),
                    "f1": float(f1.item()) if hasattr(f1, 'item') else float(f1)
                }
                per_item_scores.append(item_score)
            
            logger.info(f"Per-item BERTScore computed successfully for {len(per_item_scores)} samples")
            return per_item_scores
            
        except ImportError as e:
            logger.error("bert-score library not found for per-item computation")
            # Return error scores for each item
            error_scores = []
            for _ in range(len(predictions)):
                error_scores.append({
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "error": "bert-score library not available"
                })
            return error_scores
            
        except Exception as e:
            logger.error(f"Error computing per-item BERTScore: {str(e)}")
            # Return error scores for each item
            error_scores = []
            for _ in range(len(predictions)):
                error_scores.append({
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "error": str(e)
                })
            return error_scores
    
    def postprocess_item(self, score: Dict[str, Any]) -> float:
        """
        Postprocess individual BERTScore.
        
        Args:
            score: Raw BERTScore dictionary for a single item
            
        Returns:
            Main BERTScore (F1 score by default) for the item
        """
        # Check for errors in individual score computation
        if "error" in score:
            logger.warning(f"BERTScore per-item computation had errors: {score['error']}")
            return 0.0
        
        # Return F1 score as the main score for individual items
        return score.get("f1", 0.0)

    def postprocess(self, score: Dict[str, Any]) -> Union[float, Dict[str, Any]]:
        """
        Postprocess the computed scores.
        
        Args:
            score: Raw computed scores
            
        Returns:
            Processed score (F1 score by default, or full dict if configured)
        """
        # Check for errors in score computation
        if "error" in score:
            logger.warning(f"BERTScore computation had errors: {score['error']}")
            return 0.0 if not self.config.get("return_full_scores", False) else score
        
        # Return F1 score by default, or full dict if requested
        return_full = self.config.get("return_full_scores", False)
        
        if return_full:
            return score
        else:
            return score.get("f1", 0.0)
    
    def get_info(self) -> Dict[str, str]:
        """Get information about this metric."""
        info = super().get_info()
        
        # Safely convert values to strings, handling None values
        info.update({
            'lang': str(self.lang),
            'model_type': str(self.model_type) if self.model_type is not None else "auto",
            'num_layers': str(self.num_layers) if self.num_layers is not None else "auto",
            'idf': str(self.idf),
            'device': str(self.device) if self.device is not None else "auto",
            'batch_size': str(self.batch_size),
            'verbose': str(self.verbose)
        })
        return info


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "es",
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to compute BERTScore directly.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        lang: Language code (e.g., 'es', 'en', 'fr')
        **kwargs: Additional arguments for BERTScoreMetric
        
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score
        
        # Compute BERTScore directly
        P, R, F1 = score(
            cands=predictions,
            refs=references,
            lang=lang,
            verbose=kwargs.get('verbose', False)
        )
        
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F1.mean().item())
        }
        
    except Exception as e:
        logger.error(f"Error in compute_bertscore: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }


def compute_bertscore_detailed(
    predictions: List[str],
    references: List[str],
    lang: str = "es",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compute BERTScore with detailed per-sample results.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        lang: Language code
        verbose: Whether to print verbose information
        
    Returns:
        Dictionary with mean scores and individual scores
    """
    try:
        from bert_score import score
        
        print(f"🎯 Computing BERTScore for {len(predictions)} samples with lang='{lang}'")
        
        P, R, F1 = score(predictions, references, lang=lang, verbose=verbose)
        
        # Mean scores
        precision_mean = float(P.mean().item())
        recall_mean = float(R.mean().item())
        f1_mean = float(F1.mean().item())
        
        print(f"🎯 BERTScore results:")
        print(f"   Precision: {precision_mean:.4f}")
        print(f"   Recall:    {recall_mean:.4f}")
        print(f"   F1-Score:  {f1_mean:.4f}")
        
        # Individual scores
        individual_scores = []
        print(f"\n📈 Scores por muestra:")
        for i, (p, r, f1) in enumerate(zip(P, R, F1)):
            p_val = float(p.item())
            r_val = float(r.item())
            f1_val = float(f1.item())
            individual_scores.append({
                "precision": p_val,
                "recall": r_val,
                "f1": f1_val
            })
            print(f"   Muestra {i+1}: P={p_val:.4f}, R={r_val:.4f}, F1={f1_val:.4f}")
        
        return {
            "precision": precision_mean,
            "recall": recall_mean,
            "f1": f1_mean,
            "individual_scores": individual_scores
        }
        
    except Exception as e:
        print(f"❌ Error computing detailed BERTScore: {e}")
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        } 