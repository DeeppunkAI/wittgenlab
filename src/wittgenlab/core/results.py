"""
Results classes for storing and managing evaluation outcomes.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from .config import EvalConfig


@dataclass
class EvalResults:
    """
    Container for evaluation results from metric computations.
    """
    
    scores: Dict[str, Any]
    task: str = "general"
    num_samples: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[EvalConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    per_item_scores: Dict[str, List[Any]] = field(default_factory=dict)  # New field for per-item scores
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.config is None:
            self.config = EvalConfig()
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the results."""
        return {
            "task": self.task,
            "num_samples": self.num_samples,
            "num_metrics": len(self.scores),
            "metrics": list(self.scores.keys()),
            "timestamp": self.timestamp.isoformat(),
            "has_per_item_scores": len(self.per_item_scores) > 0
        }
    
    def get_score(self, metric: str) -> Any:
        """Get score for a specific metric."""
        return self.scores.get(metric)
    
    def get_per_item_scores(self, metric: str) -> Optional[List[Any]]:
        """Get per-item scores for a specific metric."""
        return self.per_item_scores.get(metric)
    
    def get_all_scores(self) -> Dict[str, Any]:
        """Get all computed scores."""
        return self.scores.copy()
    
    def get_all_per_item_scores(self) -> Dict[str, List[Any]]:
        """Get all per-item scores."""
        return self.per_item_scores.copy()
    
    def get_item_score(self, metric: str, item_index: int) -> Any:
        """Get score for a specific metric and item index."""
        per_item = self.per_item_scores.get(metric)
        if per_item and 0 <= item_index < len(per_item):
            return per_item[item_index]
        return None
    
    def add_per_item_scores(self, metric: str, scores: List[Any]):
        """Add per-item scores for a metric."""
        self.per_item_scores[metric] = scores
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "scores": self.scores,
            "task": self.task,
            "num_samples": self.num_samples,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "per_item_scores": self.per_item_scores,
            "summary": self.summary
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: str):
        """Save results to file."""
        path = Path(filepath)
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                f.write(self.to_json())
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> "EvalResults":
        """Load results from file."""
        path = Path(filepath)
        
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert timestamp back to datetime
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [f"EvalResults for {self.task}:"]
        lines.append(f"  Samples: {self.num_samples}")
        lines.append(f"  Metrics: {len(self.scores)}")
        
        if self.per_item_scores:
            lines.append(f"  Per-item scores available: {list(self.per_item_scores.keys())}")
        
        for metric, score in self.scores.items():
            if score is not None:
                if isinstance(score, float):
                    lines.append(f"    {metric}: {score:.4f}")
                else:
                    lines.append(f"    {metric}: {score}")
                    
                # Show per-item score statistics if available
                per_item = self.per_item_scores.get(metric)
                if per_item:
                    numeric_scores = [s for s in per_item if isinstance(s, (int, float))]
                    if numeric_scores:
                        min_score = min(numeric_scores)
                        max_score = max(numeric_scores)
                        lines.append(f"      Per-item range: {min_score:.4f} - {max_score:.4f}")
            else:
                lines.append(f"    {metric}: ERROR")
        
        return "\n".join(lines)


@dataclass
class BenchmarkResults:
    """
    Container for benchmark evaluation results.
    """
    
    scores: Dict[str, Any]
    model_name: str = "unknown"
    few_shot: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[EvalConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.config is None:
            self.config = EvalConfig()
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the benchmark results."""
        return {
            "model_name": self.model_name,
            "few_shot": self.few_shot,
            "num_benchmarks": len(self.scores),
            "benchmarks": list(self.scores.keys()),
            "timestamp": self.timestamp.isoformat()
        }
    
    def get_score(self, benchmark: str) -> Any:
        """Get score for a specific benchmark."""
        return self.scores.get(benchmark)
    
    def get_all_scores(self) -> Dict[str, Any]:
        """Get all benchmark scores."""
        return self.scores.copy()
    
    def get_average_score(self) -> Optional[float]:
        """Get average score across all benchmarks (if numeric)."""
        numeric_scores = []
        for score in self.scores.values():
            if isinstance(score, (int, float)) and score is not None:
                numeric_scores.append(score)
        
        return sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "scores": self.scores,
            "model_name": self.model_name,
            "few_shot": self.few_shot,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "summary": self.summary,
            "average_score": self.get_average_score()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert results to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def save(self, filepath: str):
        """Save results to file."""
        path = Path(filepath)
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                f.write(self.to_json())
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @classmethod
    def load(cls, filepath: str) -> "BenchmarkResults":
        """Load results from file."""
        path = Path(filepath)
        
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Convert timestamp back to datetime
        if "timestamp" in data:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        # Remove computed fields
        data.pop("summary", None)
        data.pop("average_score", None)
        
        return cls(**data)
    
    def __str__(self) -> str:
        """String representation of results."""
        lines = [f"BenchmarkResults for {self.model_name}:"]
        lines.append(f"  Few-shot: {self.few_shot}")
        lines.append(f"  Benchmarks: {len(self.scores)}")
        
        avg_score = self.get_average_score()
        if avg_score is not None:
            lines.append(f"  Average: {avg_score:.4f}")
        
        for benchmark, score in self.scores.items():
            if score is not None:
                if isinstance(score, float):
                    lines.append(f"    {benchmark}: {score:.4f}")
                else:
                    lines.append(f"    {benchmark}: {score}")
            else:
                lines.append(f"    {benchmark}: ERROR")
        
        return "\n".join(lines)


@dataclass
class ComparisonResults:
    """
    Container for model comparison results.
    """
    
    model_results: Dict[str, BenchmarkResults]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def model_names(self) -> List[str]:
        """Get list of model names."""
        return list(self.model_results.keys())
    
    @property
    def benchmarks(self) -> List[str]:
        """Get list of benchmarks (assuming all models have same benchmarks)."""
        if not self.model_results:
            return []
        first_model = next(iter(self.model_results.values()))
        return list(first_model.scores.keys())
    
    def get_ranking(self, benchmark: str) -> List[tuple]:
        """Get model ranking for a specific benchmark."""
        scores = []
        for model_name, results in self.model_results.items():
            score = results.get_score(benchmark)
            if score is not None and isinstance(score, (int, float)):
                scores.append((model_name, score))
        
        # Sort by score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)
    
    def get_overall_ranking(self) -> List[tuple]:
        """Get overall model ranking based on average scores."""
        avg_scores = []
        for model_name, results in self.model_results.items():
            avg_score = results.get_average_score()
            if avg_score is not None:
                avg_scores.append((model_name, avg_score))
        
        return sorted(avg_scores, key=lambda x: x[1], reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison results to dictionary."""
        return {
            "model_results": {
                name: results.to_dict() 
                for name, results in self.model_results.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "overall_ranking": self.get_overall_ranking()
        }
    
    def __str__(self) -> str:
        """String representation of comparison results."""
        lines = [f"Model Comparison Results ({len(self.model_results)} models):"]
        
        overall_ranking = self.get_overall_ranking()
        if overall_ranking:
            lines.append("\nOverall Ranking:")
            for i, (model_name, score) in enumerate(overall_ranking, 1):
                lines.append(f"  {i}. {model_name}: {score:.4f}")
        
        return "\n".join(lines) 