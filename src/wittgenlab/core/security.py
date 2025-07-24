"""
Security utilities for the WittgenLab evaluation framework.

This module provides functions to handle sensitive data like API keys,
tokens, and other credentials in a secure manner.
"""

import re
from typing import Any, Dict, List, Set, Union


# Common patterns for sensitive field names
SENSITIVE_PATTERNS = {
    'api_key', 'apikey', 'token', 'secret', 'password', 'auth', 'credential',
    'private_key', 'access_token', 'refresh_token', 'bearer_token',
    'openai_api_key', 'huggingface_token', 'anthropic_api_key',
    'google_api_key', 'azure_api_key', 'aws_access_key', 'aws_secret_key'
}


def is_sensitive_field(field_name: str) -> bool:
    """
    Check if a field name indicates sensitive information.
    
    Args:
        field_name: The field name to check
        
    Returns:
        True if the field is likely to contain sensitive data
    """
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in SENSITIVE_PATTERNS)


def is_sensitive_value(value: str) -> bool:
    """
    Check if a value looks like sensitive data based on patterns.
    
    Args:
        value: The value to check
        
    Returns:
        True if the value looks like an API key, token, etc.
    """
    if not isinstance(value, str) or len(value) < 10:
        return False
    
    # Common API key patterns
    patterns = [
        r'^sk-[a-zA-Z0-9]{40,}$',  # OpenAI API keys
        r'^hf_[a-zA-Z0-9]{30,}$',  # HuggingFace tokens
        r'^xoxb-[a-zA-Z0-9-]{50,}$',  # Slack bot tokens
        r'^ghp_[a-zA-Z0-9]{36}$',  # GitHub personal access tokens
        r'^[a-zA-Z0-9]{32,128}$',  # Generic long alphanumeric strings
    ]
    
    return any(re.match(pattern, value) for pattern in patterns)


def mask_sensitive_value(value: Any, mask_char: str = "*") -> str:
    """
    Mask a sensitive value for safe display.
    
    Args:
        value: The value to mask
        mask_char: Character to use for masking
        
    Returns:
        Masked version of the value
    """
    if value is None:
        return None
    
    str_value = str(value)
    
    # If very short, just mask completely
    if len(str_value) <= 8:
        return mask_char * 3
    
    # Show first 3 and last 3 characters
    return f"{str_value[:3]}{mask_char * (len(str_value) - 6)}{str_value[-3:]}"


def sanitize_dict(data: Dict[str, Any], recursive: bool = True) -> Dict[str, Any]:
    """
    Sanitize a dictionary by masking sensitive values.
    
    Args:
        data: Dictionary to sanitize
        recursive: Whether to recursively sanitize nested dictionaries
        
    Returns:
        Sanitized dictionary with sensitive values masked
    """
    sanitized = {}
    
    for key, value in data.items():
        if is_sensitive_field(key):
            sanitized[key] = mask_sensitive_value(value)
        elif isinstance(value, str) and is_sensitive_value(value):
            sanitized[key] = mask_sensitive_value(value)
        elif recursive and isinstance(value, dict):
            sanitized[key] = sanitize_dict(value, recursive=True)
        elif recursive and isinstance(value, list):
            sanitized[key] = sanitize_list(value, recursive=True)
        else:
            sanitized[key] = value
    
    return sanitized


def sanitize_list(data: List[Any], recursive: bool = True) -> List[Any]:
    """
    Sanitize a list by masking sensitive values.
    
    Args:
        data: List to sanitize
        recursive: Whether to recursively sanitize nested structures
        
    Returns:
        Sanitized list with sensitive values masked
    """
    sanitized = []
    
    for item in data:
        if isinstance(item, str) and is_sensitive_value(item):
            sanitized.append(mask_sensitive_value(item))
        elif recursive and isinstance(item, dict):
            sanitized.append(sanitize_dict(item, recursive=True))
        elif recursive and isinstance(item, list):
            sanitized.append(sanitize_list(item, recursive=True))
        else:
            sanitized.append(item)
    
    return sanitized


def sanitize_any(data: Any, recursive: bool = True) -> Any:
    """
    Sanitize any data structure by masking sensitive values.
    
    Args:
        data: Data to sanitize
        recursive: Whether to recursively sanitize nested structures
        
    Returns:
        Sanitized data with sensitive values masked
    """
    if isinstance(data, dict):
        return sanitize_dict(data, recursive=recursive)
    elif isinstance(data, list):
        return sanitize_list(data, recursive=recursive)
    elif isinstance(data, str) and is_sensitive_value(data):
        return mask_sensitive_value(data)
    else:
        return data


def create_safe_config_dict(config_obj: Any) -> Dict[str, Any]:
    """
    Create a safe dictionary representation of a configuration object.
    
    Args:
        config_obj: Configuration object to convert
        
    Returns:
        Safe dictionary with sensitive data masked
    """
    if hasattr(config_obj, 'to_safe_dict'):
        return config_obj.to_safe_dict()
    elif hasattr(config_obj, '__dict__'):
        return sanitize_dict(config_obj.__dict__)
    else:
        # Try to convert to dict if possible
        try:
            from dataclasses import asdict
            data = asdict(config_obj)
            return sanitize_dict(data)
        except (TypeError, ValueError):
            # Fallback: return string representation
            return {"config_str": str(config_obj)}


class SecureLogger:
    """Logger wrapper that automatically sanitizes sensitive data."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def _sanitize_args(self, args):
        """Sanitize logging arguments."""
        return tuple(sanitize_any(arg, recursive=False) for arg in args)
    
    def _sanitize_kwargs(self, kwargs):
        """Sanitize logging keyword arguments."""
        return sanitize_dict(kwargs, recursive=False)
    
    def debug(self, msg, *args, **kwargs):
        args = self._sanitize_args(args)
        kwargs = self._sanitize_kwargs(kwargs)
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        args = self._sanitize_args(args)
        kwargs = self._sanitize_kwargs(kwargs)
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        args = self._sanitize_args(args)
        kwargs = self._sanitize_kwargs(kwargs)
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        args = self._sanitize_args(args)
        kwargs = self._sanitize_kwargs(kwargs)
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        args = self._sanitize_args(args)
        kwargs = self._sanitize_kwargs(kwargs)
        self.logger.critical(msg, *args, **kwargs) 