"""
Configuration management for the WittgenLab evaluation framework.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Set
from pathlib import Path
import re


@dataclass
class EvalConfig:
    """
    Configuration class for the evaluation framework.
    
    This class manages all configuration options for metrics, benchmarks,
    and evaluation settings with built-in security for sensitive data.
    """
    
    # Logging configuration
    # Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
    # To ignore logs, set to "CRITICAL" or "ERROR" to suppress lower-level messages
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Cache configuration
    cache_dir: str = field(default_factory=lambda: str(Path.home() / ".wittgenlab" / "cache"))
    use_cache: bool = True
    cache_metrics: bool = True
    cache_benchmarks: bool = True
    
    # Evaluation settings
    batch_size: int = 32
    max_workers: int = 4
    timeout: int = 300  # seconds
    
    # Model settings
    max_tokens: int = 2048
    temperature: float = 0.0
    
    # Metric-specific settings
    metric_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Benchmark-specific settings
    benchmark_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # API keys and authentication
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    huggingface_token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    anthropic_api_key: Optional[str] = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    azure_api_key: Optional[str] = field(default_factory=lambda: os.getenv("AZURE_API_KEY"))
    
    # Output settings
    output_dir: str = field(default_factory=lambda: str(Path.cwd() / "eval_results"))
    save_predictions: bool = True
    save_intermediate: bool = False
    
    # Security settings
    _sensitive_fields: Set[str] = field(default_factory=lambda: {
        'openai_api_key', 'huggingface_token', 'anthropic_api_key', 
        'google_api_key', 'azure_api_key', 'api_key', 'token', 
        'password', 'secret', 'auth_token', 'access_token'
    }, init=False, repr=False)
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging level
        if isinstance(self.log_level, str):
            self.log_level = getattr(logging, self.log_level.upper())
    
    def _is_sensitive_field(self, field_name: str) -> bool:
        """Check if a field contains sensitive information."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self._sensitive_fields)
    
    def _mask_sensitive_value(self, value: Any) -> str:
        """Mask sensitive values for display."""
        if value is None:
            return None
        
        str_value = str(value)
        if len(str_value) <= 8:
            return "***"
        else:
            # Show first 3 and last 3 characters
            return f"{str_value[:3]}...{str_value[-3:]}"
    
    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively filter sensitive data from a dictionary."""
        filtered = {}
        
        for key, value in data.items():
            if self._is_sensitive_field(key):
                filtered[key] = self._mask_sensitive_value(value)
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        
        return filtered
    
    def to_safe_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with sensitive information masked."""
        from dataclasses import asdict
        
        data = asdict(self)
        # Remove internal fields
        data.pop('_sensitive_fields', None)
        
        return self._filter_sensitive_data(data)
    
    def get_metric_config(self, metric_name: str) -> Dict[str, Any]:
        """Get configuration for a specific metric."""
        return self.metric_configs.get(metric_name, {})
    
    def get_benchmark_config(self, benchmark_name: str) -> Dict[str, Any]:
        """Get configuration for a specific benchmark."""
        return self.benchmark_configs.get(benchmark_name, {})
    
    def set_metric_config(self, metric_name: str, config: Dict[str, Any]):
        """Set configuration for a specific metric."""
        self.metric_configs[metric_name] = config
    
    def set_benchmark_config(self, benchmark_name: str, config: Dict[str, Any]):
        """Set configuration for a specific benchmark."""
        self.benchmark_configs[benchmark_name] = config
    
    @classmethod
    def from_file(cls, config_path: str) -> "EvalConfig":
        """Load configuration from a file."""
        import json
        import yaml
        
        path = Path(config_path)
        
        if path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
        
        return cls(**data)
    
    def to_file(self, config_path: str, include_sensitive: bool = False):
        """
        Save configuration to a file.
        
        Args:
            config_path: Path to save the configuration
            include_sensitive: Whether to include sensitive data (default: False for security)
        """
        import json
        import yaml
        from dataclasses import asdict
        
        path = Path(config_path)
        
        if include_sensitive:
            data = asdict(self)
            data.pop('_sensitive_fields', None)
        else:
            data = self.to_safe_dict()
        
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def __repr__(self) -> str:
        """Safe string representation that masks sensitive data."""
        safe_dict = self.to_safe_dict()
        return f"EvalConfig({safe_dict})"
    
    def __str__(self) -> str:
        """Safe string representation for display."""
        safe_dict = self.to_safe_dict()
        lines = ["EvalConfig:"]
        for key, value in safe_dict.items():
            if key not in ['metric_configs', 'benchmark_configs']:  # Skip large nested configs
                lines.append(f"  {key}: {value}")
        return "\n".join(lines) 