"""
Utility functions for the Portable AI Agent
"""

import os
import json
import hashlib
import logging
import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import gzip


class Config:
    """Configuration management for the AI agent"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "ai_engine": {
                "vocab_size": 10000,
                "model_dim": 256,
                "num_heads": 8,
                "num_layers": 4,
                "max_seq_len": 512,
                "learning_rate": 1e-4,
                "adaptation_factor": 0.01
            },
            "knowledge_base": {
                "embedding_model": "all-MiniLM-L6-v2",
                "cache_size": 1000,
                "max_search_results": 10,
                "similarity_threshold": 0.1
            },
            "memory": {
                "max_context_length": 10000,
                "context_window": 50,
                "cleanup_days": 90,
                "auto_summary": True
            },
            "interface": {
                "web_host": "127.0.0.1",
                "web_port": 5000,
                "auto_save_interval": 300,
                "max_message_length": 5000
            },
            "privacy": {
                "encrypt_storage": False,
                "anonymize_logs": True,
                "data_retention_days": 365
            },
            "performance": {
                "max_memory_usage": 1024,  # MB
                "cpu_cores": -1,  # -1 for auto-detect
                "batch_size": 32,
                "enable_gpu": False
            }
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                return self.merge_configs(self.default_config, config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        
        # Create default config file
        self.save_config(self.default_config)
        return self.default_config.copy()
    
    def merge_configs(self, default: Dict, user: Dict) -> Dict:
        """Recursively merge user config with defaults"""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        config_to_save = config or self.config
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot notation path"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value):
        """Set configuration value by dot notation path"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.save_config()


class Logger:
    """Logging utility for the AI agent"""
    
    def __init__(self, name: str = "portable_ai", log_dir: str = "logs", 
                 log_level: str = "INFO", max_file_size: int = 10*1024*1024):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            # File handler with rotation
            log_file = self.log_dir / f"{name}.log"
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=max_file_size, backupCount=5
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)


class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'response_times': [],
            'learning_events': 0,
            'total_interactions': 0
        }
    
    def record_response_time(self, start_time: float):
        """Record response time for an interaction"""
        response_time = time.time() - start_time
        self.metrics['response_times'].append(response_time)
        
        # Keep only recent measurements
        if len(self.metrics['response_times']) > 1000:
            self.metrics['response_times'] = self.metrics['response_times'][-500:]
        
        return response_time
    
    def record_system_metrics(self):
        """Record current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_info.percent)
            
            # Keep only recent measurements
            for metric in ['cpu_usage', 'memory_usage']:
                if len(self.metrics[metric]) > 100:
                    self.metrics[metric] = self.metrics[metric][-50:]
                    
        except Exception as e:
            print(f"Warning: Could not record system metrics: {e}")
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        stats = {
            'uptime_seconds': time.time() - self.start_time,
            'total_interactions': self.metrics['total_interactions'],
            'learning_events': self.metrics['learning_events']
        }
        
        if self.metrics['response_times']:
            response_times = self.metrics['response_times']
            stats.update({
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            })
        
        if self.metrics['cpu_usage']:
            stats['avg_cpu_usage'] = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])
        
        if self.metrics['memory_usage']:
            stats['avg_memory_usage'] = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
        
        return stats
    
    def increment_interactions(self):
        """Increment interaction counter"""
        self.metrics['total_interactions'] += 1
    
    def increment_learning_events(self):
        """Increment learning event counter"""
        self.metrics['learning_events'] += 1


class DataManager:
    """Utility for data serialization, compression, and management"""
    
    @staticmethod
    def save_compressed(data: Any, filepath: str):
        """Save data with compression"""
        try:
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Error saving compressed data: {e}")
            raise
    
    @staticmethod
    def load_compressed(filepath: str) -> Any:
        """Load compressed data"""
        try:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading compressed data: {e}")
            raise
    
    @staticmethod
    def get_file_hash(filepath: str) -> str:
        """Get SHA256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error computing file hash: {e}")
            return ""
    
    @staticmethod
    def clean_old_files(directory: str, days_old: int = 30, pattern: str = "*"):
        """Clean up old files in a directory"""
        try:
            directory_path = Path(directory)
            if not directory_path.exists():
                return
            
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            
            for file_path in directory_path.glob(pattern):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    print(f"Cleaned up old file: {file_path}")
                    
        except Exception as e:
            print(f"Error cleaning old files: {e}")
    
    @staticmethod
    def get_directory_size(directory: str) -> int:
        """Get total size of directory in bytes"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        except Exception as e:
            print(f"Error calculating directory size: {e}")
            return 0


class TextProcessor:
    """Text processing utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        return text.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text (simple implementation)"""
        # Simple keyword extraction - in practice, use NLP libraries
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Filter and count words
        word_count = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:max_keywords]]
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate simple text similarity (Jaccard similarity)"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)


class Validator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_text_input(text: str, min_length: int = 1, max_length: int = 5000) -> bool:
        """Validate text input parameters"""
        if not isinstance(text, str):
            return False
        
        text = text.strip()
        return min_length <= len(text) <= max_length
    
    @staticmethod
    def validate_score(score: float, min_score: float = 0.0, max_score: float = 1.0) -> bool:
        """Validate numeric score"""
        try:
            score = float(score)
            return min_score <= score <= max_score
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:252] + "..."
        
        return filename


# Global instances for easy access
config = Config()
logger = Logger()
performance_monitor = PerformanceMonitor()
data_manager = DataManager()
text_processor = TextProcessor()
validator = Validator()
