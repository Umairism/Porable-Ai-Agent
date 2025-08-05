"""
Configuration management utilities
"""

import json
import os
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """
    Manages configuration for the Portable AI Agent
    """
    
    DEFAULT_CONFIG = {
        "model_path": "models/",
        "knowledge_db_path": "knowledge/knowledge.db",
        "conversation_db_path": "memory/conversations.db",
        "embedding_dim": 384,
        "knowledge_cache_size": 10000,
        "memory_cache_size": 5000,
        "memory_batch_size": 25,
        "knowledge_search_limit": 5,
        "context_history_limit": 10,
        "logging": {
            "level": "INFO",
            "file": "logs/portable_ai.log"
        },
        "ai_engine": {
            "model_name": "microsoft/DialoGPT-small",
            "max_length": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "learning_rate": 1e-4,
            "batch_size": 16
        },
        "knowledge_base": {
            "similarity_threshold": 0.7,
            "max_results": 10,
            "auto_save_interval": 300
        },
        "memory": {
            "max_context_length": 2048,
            "importance_threshold": 0.5,
            "auto_cleanup_days": 30
        },
        "web_interface": {
            "host": "127.0.0.1",
            "port": 5000,
            "debug": False
        },
        "performance": {
            "enable_caching": True,
            "cache_ttl": 3600,
            "async_processing": True,
            "batch_processing": True
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults (in case new settings were added)
                config = self.DEFAULT_CONFIG.copy()
                config.update(loaded_config)
                
                # Save the merged config back
                self.save_config(config)
                
                return config
                
            except Exception as e:
                logging.warning(f"Error loading config from {self.config_path}: {e}")
                logging.info("Using default configuration")
                
        # Create default config file
        config = self.DEFAULT_CONFIG.copy()
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to file
        """
        if config is None:
            config = self.config
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else '.', exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Error saving config to {self.config_path}: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration
        """
        return self.config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key (supports dot notation)
        
        Example:
            config.get('ai_engine.model_name')
            config.get('logging.level')
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, save: bool = True):
        """
        Set a configuration value by key (supports dot notation)
        
        Example:
            config.set('ai_engine.temperature', 0.8)
            config.set('logging.level', 'DEBUG')
        """
        keys = key.split('.')
        target = self.config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in target or not isinstance(target[k], dict):
                target[k] = {}
            target = target[k]
        
        # Set the value
        target[keys[-1]] = value
        
        if save:
            self.save_config()
    
    def update(self, updates: Dict[str, Any], save: bool = True):
        """
        Update multiple configuration values
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
        
        if save:
            self.save_config()
    
    def reset_to_defaults(self, save: bool = True):
        """
        Reset configuration to default values
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if save:
            self.save_config()
    
    def validate_config(self) -> list:
        """
        Validate configuration and return list of issues
        """
        issues = []
        
        # Check required paths
        paths_to_check = [
            self.get('model_path'),
            os.path.dirname(self.get('knowledge_db_path')),
            os.path.dirname(self.get('conversation_db_path')),
            os.path.dirname(self.get('logging.file'))
        ]
        
        for path in paths_to_check:
            if path and not os.path.exists(path):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {path}: {e}")
        
        # Check numeric values
        numeric_checks = [
            ('embedding_dim', int, 1, 2048),
            ('knowledge_cache_size', int, 100, 100000),
            ('memory_cache_size', int, 100, 50000),
            ('ai_engine.temperature', float, 0.1, 2.0),
            ('ai_engine.top_p', float, 0.1, 1.0),
            ('web_interface.port', int, 1000, 65535)
        ]
        
        for key, data_type, min_val, max_val in numeric_checks:
            value = self.get(key)
            if value is not None:
                if not isinstance(value, data_type):
                    issues.append(f"{key} must be {data_type.__name__}")
                elif not (min_val <= value <= max_val):
                    issues.append(f"{key} must be between {min_val} and {max_val}")
        
        return issues


# Convenience function for quick access
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Quick function to load configuration
    """
    manager = ConfigManager(config_path)
    return manager.get_config()


if __name__ == "__main__":
    # Test the config manager
    config_manager = ConfigManager("test_config.json")
    
    print("📋 Current configuration:")
    for key, value in config_manager.get_config().items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Validating configuration...")
    issues = config_manager.validate_config()
    
    if issues:
        print("❌ Configuration issues found:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Configuration is valid!")
    
    # Clean up test file
    if os.path.exists("test_config.json"):
        os.remove("test_config.json")
