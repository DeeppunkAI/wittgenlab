"""
Main evaluator class for the WittgenLab evaluation framework.
"""

from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from .config import EvalConfig
from .results import EvalResults, BenchmarkResults
from .security import SecureLogger, sanitize_any
from ..metrics.registry import MetricsRegistry
from ..benchmarks.registry import BenchmarksRegistry

# Use secure logger that automatically sanitizes sensitive data
base_logger = logging.getLogger(__name__)
logger = SecureLogger(base_logger)


class EvalHub:
    """
    Main evaluation hub for running metrics and benchmarks on AI models.
    
    This class provides a unified interface for:
    - Running multiple evaluation metrics
    - Executing standardized benchmarks
    - Comparing model outputs
    - Generating comprehensive reports
    """
    
    def __init__(self, config: Optional[EvalConfig] = None):
        """
        Initialize the evaluation hub.
        
        Args:
            config: Optional configuration object. If None, uses default config.
        """
        self.config = config or EvalConfig()
        self.metrics_registry = MetricsRegistry()
        self.benchmarks_registry = BenchmarksRegistry()
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("EvalHub initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metrics: List[str],
        task: str = "general",
        include_per_item: bool = False,
        **kwargs
    ) -> EvalResults:
        """
        Evaluate predictions against references using specified metrics.
        
        Args:
            predictions: List of model predictions/outputs
            references: List of ground truth references
            metrics: List of metric names to compute
            task: Task type (e.g., 'summarization', 'translation', 'generation')
            include_per_item: Whether to include per-item scores for each prediction-reference pair
            **kwargs: Additional parameters for specific metrics
            
        Returns:
            EvalResults object containing all computed metrics
        """
        logger.info(f"Starting evaluation with {len(metrics)} metrics for task: {task}")
        
        # Validate inputs
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        results = {}
        per_item_results = {}
        
        for metric_name in metrics:
            try:
                logger.info(f"Computing {metric_name}...")
                metric_class = self.metrics_registry.get_metric(metric_name)
                metric = metric_class(**kwargs)
                
                if include_per_item and hasattr(metric, 'compute_with_per_item'):
                    # Compute both overall and per-item scores
                    overall_score, per_item_scores = metric.compute_with_per_item(predictions, references)
                    results[metric_name] = overall_score
                    per_item_results[metric_name] = per_item_scores
                    
                    logger.info(f"{metric_name}: {overall_score} (per-item: {len(per_item_scores)} scores)")
                else:
                    # Compute only overall score
                    score = metric.compute(predictions, references)
                    results[metric_name] = score
                    
                    logger.info(f"{metric_name}: {score}")
                
            except Exception as e:
                logger.error(f"Error computing {metric_name}: {str(e)}")
                results[metric_name] = None
                if include_per_item:
                    per_item_results[metric_name] = [None] * len(predictions)
        
        eval_results = EvalResults(
            scores=results,
            task=task,
            num_samples=len(predictions),
            config=self.config
        )
        
        # Add per-item scores if computed
        if per_item_results:
            for metric_name, per_item_scores in per_item_results.items():
                eval_results.add_per_item_scores(metric_name, per_item_scores)
        
        return eval_results
    
    def benchmark(
        self,
        model: Any,
        benchmarks: List[str],
        few_shot: int = 0,
        batch_size: int = 1,
        **kwargs
    ) -> BenchmarkResults:
        """
        Run standardized benchmarks on a model.
        
        Args:
            model: The model to evaluate (can be HuggingFace model, API client, etc.)
            benchmarks: List of benchmark names to run
            few_shot: Number of few-shot examples to use
            batch_size: Batch size for evaluation
            **kwargs: Additional parameters for specific benchmarks
            
        Returns:
            BenchmarkResults object containing benchmark scores
        """
        logger.info(f"Starting benchmark evaluation with {len(benchmarks)} benchmarks")
        
        results = {}
        
        for benchmark_name in benchmarks:
            try:
                logger.info(f"Running {benchmark_name} benchmark...")
                benchmark_class = self.benchmarks_registry.get_benchmark(benchmark_name)
                benchmark = benchmark_class(
                    few_shot=few_shot,
                    batch_size=batch_size,
                    **kwargs
                )
                
                score = benchmark.evaluate(model)
                results[benchmark_name] = score
                
                logger.info(f"{benchmark_name}: {score}")
                
            except Exception as e:
                logger.error(f"Error running {benchmark_name}: {str(e)}")
                results[benchmark_name] = None
        
        return BenchmarkResults(
            scores=results,
            model_name=getattr(model, 'name', str(type(model))),
            few_shot=few_shot,
            config=self.config
        )
    
    def compare_models(
        self,
        models: Dict[str, Any],
        benchmarks: List[str],
        metrics: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, BenchmarkResults]:
        """
        Compare multiple models on the same benchmarks.
        
        Args:
            models: Dictionary mapping model names to model objects
            benchmarks: List of benchmark names to run
            metrics: Optional list of additional metrics to compute
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping model names to their BenchmarkResults
        """
        logger.info(f"Comparing {len(models)} models on {len(benchmarks)} benchmarks")
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            results[model_name] = self.benchmark(
                model=model,
                benchmarks=benchmarks,
                **kwargs
            )
        
        return results
    
    def judge(
        self,
        predictions: List[str],
        criteria: List[str],
        judge_models: List[str],
        consensus_method: str = "majority_vote",
        references: Optional[List[str]] = None,
        context: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate predictions using LLM-as-a-Judge with multiple models and criteria.
        
        Args:
            predictions: List of texts to evaluate
            criteria: List of evaluation criteria (e.g., ['accuracy', 'helpfulness', 'safety'])
            judge_models: List of LLM model names to use as judges
            consensus_method: Method for combining multiple judge results ('majority_vote', 'average', 'weighted_average')
            references: Optional reference texts for comparison
            context: Optional context for each prediction
            **kwargs: Additional parameters for judge configuration
            
        Returns:
            Dictionary containing judge results for each criterion
        """
        from ..judges import MultiModelJudge
        
        logger.info(f"Starting judge evaluation with {len(judge_models)} models on {len(criteria)} criteria")
        
        # Prepare model configurations
        model_configs = []
        for model_name in judge_models:
            config = {"model_name": model_name}
            config.update(kwargs)
            model_configs.append(config)
        
        # Initialize multi-model judge
        multi_judge = MultiModelJudge(
            model_configs=model_configs,
            consensus_method=consensus_method
        )
        
        # Results container
        judge_results = {}
        
        # Evaluate each criterion
        for criterion in criteria:
            logger.info(f"Evaluating criterion: {criterion}")
            criterion_results = []
            
            # Evaluate each prediction
            for i, prediction in enumerate(predictions):
                ref = references[i] if references else None
                ctx = context[i] if context else None
                
                try:
                    result = multi_judge.evaluate(
                        prediction=prediction,
                        reference=ref,
                        criterion=criterion,
                        context=ctx
                    )
                    criterion_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating prediction {i} for criterion {criterion}: {str(e)}")
                    # Create error result
                    from ..judges import JudgeResult
                    error_result = JudgeResult(
                        score=0,
                        justification=f"Error during evaluation: {str(e)}",
                        criterion=criterion,
                        model_name="error",
                        metadata={"error": True}
                    )
                    criterion_results.append(error_result)
            
            judge_results[criterion] = criterion_results
        
        # Calculate summary statistics
        summary = {}
        for criterion, results in judge_results.items():
            scores = [r.score for r in results if not r.metadata.get("error", False)]
            if scores:
                summary[criterion] = {
                    "mean_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "max_score": max(scores),
                    "num_evaluated": len(scores),
                    "num_errors": len([r for r in results if r.metadata.get("error", False)])
                }
            else:
                summary[criterion] = {
                    "mean_score": 0.0,
                    "min_score": 0,
                    "max_score": 0,
                    "num_evaluated": 0,
                    "num_errors": len(results)
                }
        
        return {
            "results": judge_results,
            "summary": summary,
            "metadata": {
                "num_predictions": len(predictions),
                "criteria": criteria,
                "judge_models": judge_models,
                "consensus_method": consensus_method
            }
        }
    
    def generate_report(
        self,
        results: Union[EvalResults, BenchmarkResults, Dict[str, BenchmarkResults]],
        output_path: Optional[str] = None,
        format: str = "html"
    ) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: Evaluation results to include in the report
            output_path: Optional path to save the report
            format: Report format ('html', 'pdf', 'json')
            
        Returns:
            Path to the generated report
        """
        from ..analysis.reports import ReportGenerator
        
        generator = ReportGenerator(format=format)
        report_path = generator.generate(results, output_path)
        
        logger.info(f"Report generated: {report_path}")
        return report_path 