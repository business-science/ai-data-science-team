"""
Model configuration management.
Provides standardized configurations for different LLM models.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from .api_keys import get_key

@dataclass
class ModelConfig:
    """
    Configuration settings for language models.
    """
    name: str
    provider: str
    api_key: str
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: Optional[list] = None
    extra_params: Optional[Dict[str, Any]] = None

class ModelConfigManager:
    """
    Manages configurations for different language models.
    """
    
    @staticmethod
    def get_openai_config(model_name: str = "gpt-4") -> ModelConfig:
        """
        Get configuration for OpenAI models.
        
        Args:
            model_name: Name of the model (e.g., "gpt-4", "gpt-3.5-turbo")
            
        Returns:
            ModelConfig with appropriate settings
        """
        base_config = {
            "gpt-4": {
                "max_tokens": 8192,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }
        
        config = base_config.get(model_name, base_config["gpt-4"])
        return ModelConfig(
            name=model_name,
            provider="openai",
            api_key=get_key("OPENAI"),
            **config
        )
    
    @staticmethod
    def get_deepseek_config(model_name: str = "deepseek-reasoner") -> ModelConfig:
        """
        Get configuration for DeepSeek models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig with appropriate settings
        """
        return ModelConfig(
            name=model_name,
            provider="deepseek",
            api_key=get_key("DEEPSEEK"),
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
    
    @staticmethod
    def get_anthropic_config(model_name: str = "claude-2") -> ModelConfig:
        """
        Get configuration for Anthropic models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            ModelConfig with appropriate settings
        """
        return ModelConfig(
            name=model_name,
            provider="anthropic",
            api_key=get_key("ANTHROPIC"),
            max_tokens=100000,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

# Create singleton instance
model_configs = ModelConfigManager()

# Convenience functions
def get_openai_config(model_name: str = "gpt-4") -> ModelConfig:
    """Get OpenAI model configuration."""
    return model_configs.get_openai_config(model_name)

def get_deepseek_config(model_name: str = "deepseek-reasoner") -> ModelConfig:
    """Get DeepSeek model configuration."""
    return model_configs.get_deepseek_config(model_name)

def get_anthropic_config(model_name: str = "claude-2") -> ModelConfig:
    """Get Anthropic model configuration."""
    return model_configs.get_anthropic_config(model_name)
