"""
Structured logging utility for ShadowAI plugins
"""
import logging
import os
from datetime import datetime
from typing import Any


class PluginLogger:
    """Structured logger for plugin operations"""
    
    def __init__(self, plugin_name: str, log_dir: str = None):
        self.plugin_name = plugin_name
        self.log_dir = log_dir or os.path.expanduser("~")
        
        # Configure file logger
        self.file_logger = logging.getLogger(f"{plugin_name}.file")
        self.file_logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = os.path.join(self.log_dir, f".{plugin_name}-debug.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # Console logger
        self.console_logger = logging.getLogger(f"{plugin_name}.console")
        self.console_logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(name)s] %(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.file_logger.debug(message)
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.file_logger.info(message)
        self.console_logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.file_logger.warning(message)
        self.console_logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.file_logger.error(message)
        self.console_logger.error(message)
    
    def log_hook_input(self, hook_name: str, input_data: Any) -> None:
        """Log hook input for debugging"""
        preview = str(input_data)[:200]
        self.debug(f"Hook '{hook_name}' invoked with input: {preview}...")
    
    def log_message(self, message_type: str, details: str = "") -> None:
        """Log message transmission"""
        self.debug(f"Sending message type={message_type}, {details}")


def create_logger(plugin_name: str, log_dir: str = None) -> PluginLogger:
    """Factory function to create plugin logger"""
    return PluginLogger(plugin_name, log_dir)
