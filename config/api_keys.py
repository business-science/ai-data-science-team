"""
API key management and access control.
Provides a clean interface for accessing API keys from the master .env file.
"""

from typing import Optional, Dict, List
from .utils.env_loader import get_api_key, get_all_keys_for_service, list_available_services

class APIKeyManager:
    """
    Manages access to API keys with additional functionality and validation.
    """
    
    @staticmethod
    def get_openai_key() -> str:
        """Get OpenAI API key."""
        return get_api_key('OPENAI')
    
    @staticmethod
    def get_deepseek_key() -> str:
        """Get DeepSeek API key."""
        return get_api_key('DEEPSEEK')
    
    @staticmethod
    def get_anthropic_key() -> str:
        """Get Anthropic API key."""
        return get_api_key('ANTHROPIC')
    
    @staticmethod
    def get_key(service: str) -> Optional[str]:
        """
        Get API key for any service.
        
        Args:
            service: Service name (e.g., 'OPENAI', 'DEEPSEEK')
            
        Returns:
            API key if found, None otherwise
        """
        return get_api_key(service)
    
    @staticmethod
    def get_all_keys(service: str) -> Dict[str, str]:
        """
        Get all keys for a service (including alternates).
        
        Args:
            service: Service name (e.g., 'OPENAI', 'DEEPSEEK')
            
        Returns:
            Dictionary of all related keys
        """
        return get_all_keys_for_service(service)
    
    @staticmethod
    def list_services() -> List[str]:
        """
        List all available services.
        
        Returns:
            List of service names with configured API keys
        """
        return list_available_services()

# Create singleton instance
api_keys = APIKeyManager()

# Convenience functions
def get_openai_key() -> str:
    """Get OpenAI API key."""
    return api_keys.get_openai_key()

def get_deepseek_key() -> str:
    """Get DeepSeek API key."""
    return api_keys.get_deepseek_key()

def get_anthropic_key() -> str:
    """Get Anthropic API key."""
    return api_keys.get_anthropic_key()

def get_key(service: str) -> Optional[str]:
    """Get API key for any service."""
    return api_keys.get_key(service)

def get_all_keys(service: str) -> Dict[str, str]:
    """Get all keys for a service."""
    return api_keys.get_all_keys(service)

def list_services() -> List[str]:
    """List all available services."""
    return api_keys.list_services()
