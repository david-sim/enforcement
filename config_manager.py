"""
Configuration management utility for the enforcement application.

This module provides centralized configuration management by loading settings
from config.json and providing easy access to configuration values.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the application."""
    
    _instance: Optional['Config'] = None
    _config_data: Dict[str, Any] = {}
    
    def __new__(cls) -> 'Config':
        """Ensure singleton pattern for configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize configuration if not already loaded."""
        if not self._config_data:
            self.load_config()
    
    def load_config(self, config_file: str = "config.json") -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_file: Path to the configuration file (default: config.json)
        """
        try:
            config_path = Path(__file__).parent / config_file
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = json.load(f)
                
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error loading configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Examples:
            config.get('llm.model')  # Returns "gpt-4.1"
            config.get('rate_limiting.rate_limit_per_hour')  # Returns "950"
        """
        keys = key_path.split('.')
        value = self._config_data
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_rate_limiting_config(self) -> Dict[str, str]:
        """Get all rate limiting configuration values."""
        return self._config_data.get('rate_limiting', {})
    
    def get_llm_config(self) -> Dict[str, str]:
        """Get all LLM configuration values."""
        return self._config_data.get('llm', {})
    
    def get_rules_config(self) -> Dict[str, str]:
        """Get all rules configuration values."""
        return self._config_data.get('rules', {})
    
    def get_industrial_occupant_rules(self) -> str:
        """Get industrial occupant rules."""
        return self.get('rules.industrial_occupant_rules', '')
    
    def get_industrial_compliance_rules(self) -> str:
        """Get industrial compliance rules."""
        return self.get('rules.industrial_compliance_rules', '')
    
    def get_shophouse_occupant_rules(self) -> str:
        """Get shophouse occupant rules."""
        return self.get('rules.shophouse_occupant_rules', '')
    
    def get_shophouse_compliance_rules(self) -> str:
        """Get shophouse compliance rules."""
        return self.get('rules.shophouse_compliance_rules', '')
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config_data.clear()
        self.load_config()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(loaded_keys={list(self._config_data.keys())})"


# Create global configuration instance
config = Config()

# Convenience functions for common access patterns
def get_rate_limit_per_hour() -> str:
    """Get rate limit per hour setting."""
    return config.get('rate_limiting.rate_limit_per_hour', '950')

def get_rate_limit_delay() -> str:
    """Get rate limit delay setting."""
    return config.get('rate_limiting.rate_limit_delay', '3.6')

def get_min_delay_between_calls() -> str:
    """Get minimum delay between calls setting."""
    return config.get('rate_limiting.min_delay_between_calls', '4.0')

def get_llm_model() -> str:
    """Get LLM model setting."""
    return config.get('llm.model', 'gpt-4.1')

def get_llm_temperature() -> str:
    """Get LLM temperature setting."""
    return config.get('llm.temperature', '0.78')

def get_industrial_occupant_rules() -> str:
    """Get industrial occupant rules."""
    return config.get_industrial_occupant_rules()

def get_industrial_compliance_rules() -> str:
    """Get industrial compliance rules."""
    return config.get_industrial_compliance_rules()

def get_shophouse_occupant_rules() -> str:
    """Get shophouse occupant rules."""
    return config.get_shophouse_occupant_rules()

def get_shophouse_compliance_rules() -> str:
    """Get shophouse compliance rules."""
    return config.get_shophouse_compliance_rules()
