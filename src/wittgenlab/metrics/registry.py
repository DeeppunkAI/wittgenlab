"""
Registry system for managing evaluation metrics.
"""

from typing import Dict, Type, List
import importlib
import logging

from .base import BaseMetric

logger = logging.getLogger(__name__)


class MetricsRegistry:
    """
    Registry for managing and accessing evaluation metrics.
    
    This class maintains a catalog of all available metrics and provides
    methods to retrieve and instantiate them.
    """
    
    def __init__(self):
        """Initialize the metrics registry."""
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize all available metrics."""
        # N-gram metrics
        self._register_ngram_metrics()
        
        # Semantic metrics
        self._register_semantic_metrics()
        
        # Perplexity metrics
        self._register_perplexity_metrics()
        
        # Custom metrics
        self._register_custom_metrics()
        
        logger.info(f"Initialized {len(self._metrics)} metrics")
    
    def _register_ngram_metrics(self):
        """Register n-gram based metrics."""
        try:
            from .ngram.bleu import BLEUMetric
            self._metrics['bleu'] = BLEUMetric
            
            from .ngram.rouge import ROUGEMetric
            self._metrics['rouge'] = ROUGEMetric
            self._metrics['rouge-1'] = ROUGEMetric
            self._metrics['rouge-2'] = ROUGEMetric
            self._metrics['rouge-l'] = ROUGEMetric
            
            from .ngram.meteor import METEORMetric
            self._metrics['meteor'] = METEORMetric
            
            from .ngram.cider import CIDErMetric
            self._metrics['cider'] = CIDErMetric
            
        except ImportError as e:
            logger.warning(f"Could not import n-gram metrics: {e}")
    
    def _register_semantic_metrics(self):
        """Register semantic similarity metrics."""
        try:
            from .semantic.bertscore import BERTScoreMetric
            self._metrics['bertscore'] = BERTScoreMetric
            
            from .semantic.moverscore import MoverScoreMetric
            self._metrics['moverscore'] = MoverScoreMetric
            
            from .semantic.bleurt import BLEURTMetric
            self._metrics['bleurt'] = BLEURTMetric
            
        except ImportError as e:
            logger.warning(f"Could not import semantic metrics: {e}")
    
    def _register_perplexity_metrics(self):
        """Register perplexity-based metrics."""
        try:
            from .perplexity.gpt_perplexity import GPTPerplexityMetric
            self._metrics['gpt_perplexity'] = GPTPerplexityMetric
            
            from .perplexity.bert_perplexity import BERTPerplexityMetric  
            self._metrics['bert_perplexity'] = BERTPerplexityMetric
            
        except ImportError as e:
            logger.warning(f"Could not import perplexity metrics: {e}")
    
    def _register_custom_metrics(self):
        """Register custom metrics."""
        try:
            from .custom.diversity import DiversityMetric
            self._metrics['diversity'] = DiversityMetric
            
            from .custom.coherence import CoherenceMetric
            self._metrics['coherence'] = CoherenceMetric
            
        except ImportError as e:
            logger.warning(f"Could not import custom metrics: {e}")
    
    def register_metric(self, name: str, metric_class: Type[BaseMetric]):
        """
        Register a new metric.
        
        Args:
            name: Name to register the metric under
            metric_class: The metric class to register
        """
        if not issubclass(metric_class, BaseMetric):
            raise ValueError("Metric class must inherit from BaseMetric")
        
        self._metrics[name] = metric_class
        logger.info(f"Registered metric: {name}")
    
    def get_metric(self, name: str) -> Type[BaseMetric]:
        """
        Get a metric class by name.
        
        Args:
            name: Name of the metric
            
        Returns:
            The metric class
            
        Raises:
            KeyError: If metric is not found
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found. Available metrics: {self.list_metrics()}")
        
        return self._metrics[name]
    
    def list_metrics(self) -> List[str]:
        """Get list of all available metric names."""
        return list(self._metrics.keys())
    
    def list_metrics_by_category(self) -> Dict[str, List[str]]:
        """Get metrics organized by category."""
        categories = {
            'ngram': [],
            'semantic': [],
            'perplexity': [],
            'custom': []
        }
        
        for name, metric_class in self._metrics.items():
            # Determine category based on module path
            module_path = metric_class.__module__
            if 'ngram' in module_path:
                categories['ngram'].append(name)
            elif 'semantic' in module_path:
                categories['semantic'].append(name)
            elif 'perplexity' in module_path:
                categories['perplexity'].append(name)
            else:
                categories['custom'].append(name)
        
        return categories
    
    def get_metric_info(self, name: str) -> Dict[str, str]:
        """
        Get information about a specific metric.
        
        Args:
            name: Name of the metric
            
        Returns:
            Dictionary with metric information
        """
        metric_class = self.get_metric(name)
        
        return {
            'name': name,
            'class': metric_class.__name__,
            'module': metric_class.__module__,
            'description': metric_class.__doc__ or "No description available",
            'category': self._get_category(metric_class)
        }
    
    def _get_category(self, metric_class: Type[BaseMetric]) -> str:
        """Determine the category of a metric class."""
        module_path = metric_class.__module__
        if 'ngram' in module_path:
            return 'ngram'
        elif 'semantic' in module_path:
            return 'semantic'
        elif 'perplexity' in module_path:
            return 'perplexity'
        else:
            return 'custom' 