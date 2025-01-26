"""
Environment variable loader with validation and logging.
Centralizes access to the master .env file.
"""

import os
from typing import Optional, Dict, List
from dotenv import load_dotenv
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to master .env file
MASTER_ENV_PATH = Path("C:/Users/chris/ai-data-science-team/.env")

class EnvLoader:
    """
    Enhanced environment variable loader with validation and logging.
    Provides centralized access to API keys and configurations.
    """
    
    def __init__(self):
        self._load_environment()
        self._validate_critical_keys()
        
    def _load_environment(self):
        """Load environment variables from master .env file."""
        if not MASTER_ENV_PATH.exists():
            raise FileNotFoundError(f"Master .env file not found at {MASTER_ENV_PATH}")
        
        load_dotenv(MASTER_ENV_PATH)
        logger.info(f"Loaded environment variables from {MASTER_ENV_PATH}")
        
    def _validate_critical_keys(self):
        """Validate that critical API keys exist."""
        critical_keys = [
            'OPENAI_API_KEY',
            'DEEPSEEK_API_KEY',
            'ANTHROPIC_API_KEY',
        ]
        
        missing_keys = [key for key in critical_keys if not os.getenv(key)]
        if missing_keys:
            logger.warning(f"Missing critical API keys: {', '.join(missing_keys)}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service.
        
        Args:
            service: Service name (e.g., 'OPENAI', 'DEEPSEEK')
            
        Returns:
            API key if found, None otherwise
        """
        key_name = f"{service.upper()}_API_KEY"
        key = os.getenv(key_name)
        
        if not key:
            logger.warning(f"API key not found for service: {service}")
            return None
            
        logger.info(f"Retrieved API key for service: {service}")
        return key
    
    def get_all_keys_for_service(self, service: str) -> Dict[str, str]:
        """
        Get all environment variables related to a service.
        
        Args:
            service: Service name (e.g., 'OPENAI', 'DEEPSEEK')
            
        Returns:
            Dictionary of all related environment variables
        """
        service_prefix = service.upper()
        return {
            key: value 
            for key, value in os.environ.items() 
            if key.startswith(service_prefix)
        }
    
    def list_available_services(self) -> List[str]:
        """
        List all services that have API keys configured.
        
        Returns:
            List of service names
        """
        services = set()
        for key in os.environ:
            if key.endswith('_API_KEY'):
                service = key.replace('_API_KEY', '')
                services.add(service)
        return sorted(list(services))

# Create singleton instance
env_loader = EnvLoader()

# Convenience functions
def get_api_key(service: str) -> Optional[str]:
    """Get API key for a service."""
    return env_loader.get_api_key(service)

def get_all_keys_for_service(service: str) -> Dict[str, str]:
    """Get all environment variables for a service."""
    return env_loader.get_all_keys_for_service(service)

def list_available_services() -> List[str]:
    """List all services with API keys."""
    return env_loader.list_available_services()
