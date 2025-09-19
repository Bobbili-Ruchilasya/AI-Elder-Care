"""
Configuration Management and Utilities
"""

import yaml
import json
import os
from typing import Dict, Any, Optional
import logging
from pathlib import Path

class Config:
    """
    Configuration management class
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_data = {}
        
        if config_path:
            self.load_config(config_path)
        else:
            self.load_default_config()
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'r') as f:
                self.config_data = yaml.safe_load(f)
        elif path.suffix.lower() == '.json':
            with open(path, 'r') as f:
                self.config_data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")
    
    def load_default_config(self):
        """Load default configuration"""
        self.config_data = {
            'model': {
                'speech_input_dim': 128,
                'text_model_name': 'bert-base-uncased',
                'hidden_dim': 256,
                'num_classes': 1,
                'dropout': 0.3
            },
            'training': {
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'batch_size': 32,
                'num_epochs': 50,
                'early_stopping_patience': 10,
                'gradient_clip_norm': 1.0
            },
            'data': {
                'speech_sample_rate': 16000,
                'text_max_length': 512,
                'normalize_features': True,
                'test_size': 0.2,
                'val_size': 0.1,
                'stratify': True,
                'apply_augmentation': False,
                'num_workers': 0
            },
            'paths': {
                'model_save_dir': 'models',
                'log_dir': 'logs',
                'cache_dir': 'cache',
                'results_dir': 'results'
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, save_path: str):
        """Save configuration to file"""
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() == '.yaml' or path.suffix.lower() == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.config_data, f, default_flow_style=False)
        elif path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(self.config_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported save format: {path.suffix}")

class Logger:
    """
    Centralized logging utility
    """
    
    def __init__(self, name: str, log_file: Optional[str] = None, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        if log_file:
            file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        if log_file:
            self.logger.addHandler(file_handler)
    
    def get_logger(self):
        return self.logger

def ensure_directory(path: str):
    """Ensure directory exists, create if not"""
    os.makedirs(path, exist_ok=True)

def save_results(results: Dict, save_path: str):
    """Save results to file"""
    ensure_directory(os.path.dirname(save_path))
    
    if save_path.endswith('.json'):
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif save_path.endswith('.yaml') or save_path.endswith('.yml'):
        with open(save_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        # Default to JSON
        with open(save_path + '.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
