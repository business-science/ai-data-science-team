"""
Global configuration management for AI Data Science Team.
Provides centralized access to API keys and model configurations.
"""

from .api_keys import (
    get_openai_key,
    get_deepseek_key,
    get_anthropic_key,
    get_key,
    get_all_keys,
    list_services
)

from .model_configs import (
    ModelConfig,
    get_openai_config,
    get_deepseek_config,
    get_anthropic_config
)

__all__ = [
    # API Keys
    'get_openai_key',
    'get_deepseek_key',
    'get_anthropic_key',
    'get_key',
    'get_all_keys',
    'list_services',
    
    # Model Configs
    'ModelConfig',
    'get_openai_config',
    'get_deepseek_config',
    'get_anthropic_config'
]
