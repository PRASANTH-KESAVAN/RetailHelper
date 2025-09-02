# src/utils/logger.py

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
import json
import traceback
from typing import Optional, Dict, Any
import functools
import time

class RetailAnalyticsLogger:
    """
    Comprehensive logging system for Retail Customer Analytics Project
    
    Features:
    - Multiple log levels and handlers
    - Rotating file logs
    - Structured logging with JSON format
    - Performance tracking
    - Error tracking with stack traces
    - User activity logging
    - Model training and prediction logging
    """
    
    def __init__(self, 
                 name: str = "retail_analytics",
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 structured_logging: bool = True):
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.structured_logging = structured_logging
        
        # Create logs directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup handlers
        if enable_console:
            self._setup_console_handler()
        
        if enable_file:
            self._setup_file_handlers(max_bytes, backup_count)
        
        # Initialize session tracking
        self.session_id = self._generate_session_id()
        
        # Log startup
        self.info("Logger initialized", extra={
            "session_id": self.session_id,
            "log_level": log_level,
            "log_dir": str(self.log_dir)
        })
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    
    def _setup_console_handler(self):
        """Setup console logging handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        if self.structured_logging:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, max_bytes: int, backup_count: int):
        """Setup file logging handlers with rotation"""
        
        # Main application log
        main_log_file = self.log_dir / f"{self.name}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        main_handler.setLevel(logging.DEBUG)
        
        # Error log (errors and critical only)
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        
        # Performance log
        performance_log_file = self.log_dir / f"{self.name}_performance.log"
        self.performance_handler = logging.handlers.RotatingFileHandler(
            performance_log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        self.performance_handler.setLevel(logging.INFO)
        
        # Setup formatters
        if self.structured_logging:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        main_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        self.performance_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        
        # Create separate performance logger
        self.performance_logger = logging.getLogger(f"{self.name}_performance")
        self.performance_logger.setLevel(logging.INFO)
        self.performance_logger.addHandler(self.performance_handler)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        self._log(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message"""
        self._log(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self._log(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False):
        """Log error message"""
        if exc_info:
            extra = extra or {}
            extra['traceback'] = traceback.format_exc()
        self._log(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message"""
        self._log(logging.CRITICAL, message, extra)
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal logging method with structured data"""
        if extra is None:
            extra = {}
        
        # Add session context
        extra['session_id'] = self.session_id
        extra['timestamp'] = datetime.now().isoformat()
        
        self.logger.log(level, message, extra=extra)
    
    def log_user_activity(self, user_id: str, activity: str, details: Optional[Dict[str, Any]] = None):
        """Log user activity for analytics"""
        extra = {
            'user_id': user_id,
            'activity_type': 'user_activity',
            'activity': activity,
            'details': details or {}
        }
        self.info(f"User activity: {activity}", extra=extra)
    
    def log_model_training(self, model_name: str, status: str, metrics: Optional[Dict[str, Any]] = None):
        """Log model training events"""
        extra = {
            'model_name': model_name,
            'activity_type': 'model_training',
            'status': status,
            'metrics': metrics or {}
        }
        self.info(f"Model training {status}: {model_name}", extra=extra)
    
    def log_model_prediction(self, model_name: str, prediction_count: int, execution_time: float):
        """Log model prediction events"""
        extra = {
            'model_name': model_name,
            'activity_type': 'model_prediction',
            'prediction_count': prediction_count,
            'execution_time_seconds': execution_time
        }
        self.info(f"Model prediction completed: {model_name}", extra=extra)
    
    def log_data_processing(self, process_type: str, record_count: int, status: str):
        """Log data processing events"""
        extra = {
            'activity_type': 'data_processing',
            'process_type': process_type,
            'record_count': record_count,
            'status': status
        }
        self.info(f"Data processing {status}: {process_type}", extra=extra)
    
    def log_dashboard_access(self, page: str, user_agent: str, ip_address: str = "unknown"):
        """Log dashboard page access"""
        extra = {
            'activity_type': 'dashboard_access',
            'page': page,
            'user_agent': user_agent,
            'ip_address': ip_address
        }
        self.info(f"Dashboard access: {page}", extra=extra)
    
    def log_performance(self, operation: str, execution_time: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        extra = {
            'activity_type': 'performance',
            'operation': operation,
            'execution_time_seconds': execution_time,
            'details': details or {}
        }
        self.performance_logger.info(f"Performance: {operation} took {execution_time:.3f}s", extra=extra)
    
    def log_system_health(self, component: str, status: str, metrics: Optional[Dict[str, Any]] = None):
        """Log system health metrics"""
        extra = {
            'activity_type': 'system_health',
            'component': component,
            'status': status,
            'metrics': metrics or {}
        }
        self.info(f"System health: {component} - {status}", extra=extra)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                             'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                             'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                             'message', 'exc_info', 'exc_text', 'stack_info']:
                    log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


def timing_decorator(logger_instance):
    """Decorator to log function execution time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger_instance.log_performance(
                    operation=f"{func.__module__}.{func.__name__}",
                    execution_time=execution_time,
                    details={'args_count': len(args), 'kwargs_count': len(kwargs)}
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger_instance.error(
                    f"Function {func.__name__} failed after {execution_time:.3f}s",
                    extra={'function': func.__name__, 'error': str(e)},
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


def error_handler_decorator(logger_instance):
    """Decorator to handle and log exceptions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger_instance.error(
                    f"Unhandled exception in {func.__name__}: {str(e)}",
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    },
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


class LoggerConfig:
    """Configuration class for logger settings"""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'log_level': 'INFO',
        'log_dir': 'logs',
        'max_bytes': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'enable_console': True,
        'enable_file': True,
        'structured_logging': True
    }
    
    @classmethod
    def from_env(cls) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = cls.DEFAULT_CONFIG.copy()
        
        # Override with environment variables if present
        config['log_level'] = os.getenv('LOG_LEVEL', config['log_level'])
        config['log_dir'] = os.getenv('LOG_DIR', config['log_dir'])
        config['max_bytes'] = int(os.getenv('LOG_MAX_BYTES', config['max_bytes']))
        config['backup_count'] = int(os.getenv('LOG_BACKUP_COUNT', config['backup_count']))
        
        # Boolean environment variables
        config['enable_console'] = os.getenv('LOG_ENABLE_CONSOLE', 'true').lower() == 'true'
        config['enable_file'] = os.getenv('LOG_ENABLE_FILE', 'true').lower() == 'true'
        config['structured_logging'] = os.getenv('LOG_STRUCTURED', 'true').lower() == 'true'
        
        return config


# Global logger instances
_loggers = {}

def get_logger(name: str = "retail_analytics", **kwargs) -> RetailAnalyticsLogger:
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        **kwargs: Additional configuration options
    
    Returns:
        RetailAnalyticsLogger instance
    """
    if name not in _loggers:
        # Load configuration from environment
        config = LoggerConfig.from_env()
        config.update(kwargs)
        
        _loggers[name] = RetailAnalyticsLogger(name=name, **config)
    
    return _loggers[name]


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> RetailAnalyticsLogger:
    """
    Setup logging for the entire application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
    
    Returns:
        Configured logger instance
    """
    return get_logger("retail_analytics", log_level=log_level, log_dir=log_dir)


# Convenience functions for common logging operations
def log_startup(logger: RetailAnalyticsLogger, app_name: str, version: str = "1.0.0"):
    """Log application startup"""
    logger.info(f"Starting {app_name}", extra={
        'app_name': app_name,
        'version': version,
        'python_version': sys.version.split()[0],
        'platform': sys.platform
    })


def log_shutdown(logger: RetailAnalyticsLogger, app_name: str):
    """Log application shutdown"""
    logger.info(f"Shutting down {app_name}", extra={'app_name': app_name})


def log_api_request(logger: RetailAnalyticsLogger, method: str, endpoint: str, 
                   status_code: int, response_time: float):
    """Log API request"""
    logger.info(f"API Request: {method} {endpoint}", extra={
        'activity_type': 'api_request',
        'method': method,
        'endpoint': endpoint,
        'status_code': status_code,
        'response_time_ms': response_time * 1000
    })


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    logger = setup_logging(log_level="DEBUG")
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    logger.info("User logged in", extra={
        'user_id': 'user123',
        'ip_address': '192.168.1.1',
        'user_agent': 'Mozilla/5.0...'
    })
    
    # Test activity logging
    logger.log_user_activity('user123', 'dashboard_access', {'page': 'home'})
    logger.log_model_training('churn_model', 'completed', {'accuracy': 0.85, 'auc': 0.92})
    logger.log_data_processing('data_cleaning', 1000, 'completed')
    
    # Test performance logging
    logger.log_performance('data_processing', 2.5, {'records_processed': 1000})
    
    # Test decorators
    @timing_decorator(logger)
    @error_handler_decorator(logger)
    def test_function(x, y):
        time.sleep(0.1)  # Simulate work
        return x + y
    
    result = test_function(5, 3)
    logger.info(f"Test function result: {result}")
    
    print("✅ Logger testing completed. Check the logs directory for output files.")