"""
Test script to verify configuration setup.
"""

import os
from . import (
    get_openai_key,
    get_deepseek_key,
    get_anthropic_key,
    list_services,
    get_openai_config
)

def test_configuration():
    """Test the configuration setup."""
    print("Testing Configuration Setup\n")
    
    print("1. Available Services:")
    services = list_services()
    print(f"Found {len(services)} services: {', '.join(services)}\n")
    
    print("2. Testing API Keys:")
    keys = {
        "OpenAI": get_openai_key(),
        "DeepSeek": get_deepseek_key(),
        "Anthropic": get_anthropic_key()
    }
    
    for service, key in keys.items():
        status = "✓ Found" if key else "✗ Missing"
        print(f"{service}: {status}")
    print()
    
    print("3. Testing Model Configuration:")
    config = get_openai_config("gpt-4")
    print(f"Model: {config.name}")
    print(f"Provider: {config.provider}")
    print(f"Max Tokens: {config.max_tokens}")
    print(f"Temperature: {config.temperature}")

if __name__ == "__main__":
    test_configuration()
