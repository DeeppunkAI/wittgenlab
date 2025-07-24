"""
Registry system for managing evaluation benchmarks.
"""

from typing import Dict, Type, List
import logging

from .base import BaseBenchmark

logger = logging.getLogger(__name__)


class BenchmarksRegistry:
    """
    Registry for managing and accessing evaluation benchmarks.
    
    This class maintains a catalog of all available benchmarks and provides
    methods to retrieve and instantiate them.
    """
    
    def __init__(self):
        """Initialize the benchmarks registry."""
        self._benchmarks: Dict[str, Type[BaseBenchmark]] = {}
        self._initialize_benchmarks()
    
    def _initialize_benchmarks(self):
        """Initialize all available benchmarks."""
        # GLUE benchmarks
        self._register_glue_benchmarks()
        
        # Knowledge benchmarks
        self._register_knowledge_benchmarks()
        
        # Code benchmarks
        self._register_code_benchmarks()
        
        # Safety benchmarks
        self._register_safety_benchmarks()
        
        # Multilingual benchmarks
        self._register_multilingual_benchmarks()
        
        logger.info(f"Initialized {len(self._benchmarks)} benchmarks")
    
    def _register_glue_benchmarks(self):
        """Register GLUE and related benchmarks."""
        # Placeholder implementations
        pass
    
    def _register_knowledge_benchmarks(self):
        """Register knowledge-based benchmarks."""
        try:
            from .knowledge.mmlu import MMLUBenchmark
            self._benchmarks['mmlu'] = MMLUBenchmark
            
            from .knowledge.arc import ARCBenchmark
            self._benchmarks['arc'] = ARCBenchmark
            
            from .knowledge.hellaswag import HellaSwagBenchmark
            self._benchmarks['hellaswag'] = HellaSwagBenchmark
            
        except ImportError as e:
            logger.warning(f"Could not import knowledge benchmarks: {e}")
    
    def _register_code_benchmarks(self):
        """Register code evaluation benchmarks."""
        try:
            from .code.humaneval import HumanEvalBenchmark
            self._benchmarks['humaneval'] = HumanEvalBenchmark
            
            from .code.mbpp import MBPPBenchmark
            self._benchmarks['mbpp'] = MBPPBenchmark
            
        except ImportError as e:
            logger.warning(f"Could not import code benchmarks: {e}")
    
    def _register_safety_benchmarks(self):
        """Register safety and alignment benchmarks."""
        try:
            from .safety.toxigen import ToxiGenBenchmark
            self._benchmarks['toxigen'] = ToxiGenBenchmark
            
            from .safety.truthfulqa import TruthfulQABenchmark
            self._benchmarks['truthfulqa'] = TruthfulQABenchmark
            
        except ImportError as e:
            logger.warning(f"Could not import safety benchmarks: {e}")
    
    def _register_multilingual_benchmarks(self):
        """Register multilingual benchmarks."""
        # Placeholder for multilingual benchmarks
        pass
    
    def register_benchmark(self, name: str, benchmark_class: Type[BaseBenchmark]):
        """
        Register a new benchmark.
        
        Args:
            name: Name to register the benchmark under
            benchmark_class: The benchmark class to register
        """
        if not issubclass(benchmark_class, BaseBenchmark):
            raise ValueError("Benchmark class must inherit from BaseBenchmark")
        
        self._benchmarks[name] = benchmark_class
        logger.info(f"Registered benchmark: {name}")
    
    def get_benchmark(self, name: str) -> Type[BaseBenchmark]:
        """
        Get a benchmark class by name.
        
        Args:
            name: Name of the benchmark
            
        Returns:
            The benchmark class
            
        Raises:
            KeyError: If benchmark is not found
        """
        if name not in self._benchmarks:
            raise KeyError(f"Benchmark '{name}' not found. Available benchmarks: {self.list_benchmarks()}")
        
        return self._benchmarks[name]
    
    def list_benchmarks(self) -> List[str]:
        """Get list of all available benchmark names."""
        return list(self._benchmarks.keys())
    
    def list_benchmarks_by_category(self) -> Dict[str, List[str]]:
        """Get benchmarks organized by category."""
        categories = {
            'glue': [],
            'knowledge': [],
            'code': [],
            'safety': [],
            'multilingual': []
        }
        
        for name, benchmark_class in self._benchmarks.items():
            # Determine category based on module path
            module_path = benchmark_class.__module__
            if 'glue' in module_path:
                categories['glue'].append(name)
            elif 'knowledge' in module_path:
                categories['knowledge'].append(name)
            elif 'code' in module_path:
                categories['code'].append(name)
            elif 'safety' in module_path:
                categories['safety'].append(name)
            elif 'multilingual' in module_path:
                categories['multilingual'].append(name)
        
        return categories
    
    def get_benchmark_info(self, name: str) -> Dict[str, str]:
        """
        Get information about a specific benchmark.
        
        Args:
            name: Name of the benchmark
            
        Returns:
            Dictionary with benchmark information
        """
        benchmark_class = self.get_benchmark(name)
        
        return {
            'name': name,
            'class': benchmark_class.__name__,
            'module': benchmark_class.__module__,
            'description': benchmark_class.__doc__ or "No description available",
            'category': self._get_category(benchmark_class)
        }
    
    def _get_category(self, benchmark_class: Type[BaseBenchmark]) -> str:
        """Determine the category of a benchmark class."""
        module_path = benchmark_class.__module__
        if 'glue' in module_path:
            return 'glue'
        elif 'knowledge' in module_path:
            return 'knowledge'
        elif 'code' in module_path:
            return 'code'
        elif 'safety' in module_path:
            return 'safety'
        elif 'multilingual' in module_path:
            return 'multilingual'
        else:
            return 'other' 