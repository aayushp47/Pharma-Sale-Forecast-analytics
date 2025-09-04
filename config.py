#!/usr/bin/env python3
"""
Configuration settings for Healthcare Sales Analysis Flask App
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'healthcare-sales-analytics-secret-key-2024'
    
    # File upload settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Database settings (for future use)
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///healthcare_sales.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session settings
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=2)
    
    # API settings
    API_RATE_LIMIT = "100 per hour"
    
    # Forecasting settings
    FORECASTING_MODELS = {
        'arima': {
            'name': 'ARIMA',
            'description': 'Autoregressive Integrated Moving Average',
            'params': {'order': (1, 1, 1), 'seasonal_order': (1, 1, 1, 12)}
        },
        'prophet': {
            'name': 'Prophet',
            'description': 'Facebook Prophet Time Series Forecasting',
            'params': {'yearly_seasonality': True, 'weekly_seasonality': False, 'daily_seasonality': False}
        },
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Gradient Boosting for Time Series',
            'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        },
        'linear': {
            'name': 'Linear Regression',
            'description': 'Linear Trend Forecasting',
            'params': {'fit_intercept': True}
        },
        'moving_avg': {
            'name': 'Moving Average',
            'description': 'Simple Moving Average',
            'params': {'window': 3}
        },
        'exponential_smoothing': {
            'name': 'Exponential Smoothing',
            'description': 'Triple Exponential Smoothing (Holt-Winters)',
            'params': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12}
        }
    }
    
    # Analytics settings
    PERFORMANCE_METRICS = ['revenue', 'units', 'growth', 'profit_margin']
    TIME_PERIODS = ['daily', 'weekly', 'monthly', 'quarterly', 'yearly']
    
    # Demo data settings
    DEMO_DATA_PARAMS = {
        'drugs': ['Aspirin', 'Ibuprofen', 'Lisinopril', 'Metformin', 'Amlodipine', 
                 'Omeprazole', 'Atorvastatin', 'Levothyroxine', 'Simvastatin', 'Warfarin'],
        'regions': ['North America', 'Europe', 'Asia-Pacific', 'Latin America'],
        'years': 4,  # Number of years of historical data
        'seasonality_factor': 0.3,  # Strength of seasonal patterns
        'trend_factor': 0.1,  # Annual growth/decline rate
        'noise_factor': 0.2  # Random noise in data
    }
    
    # Insights generation settings
    INSIGHTS_CONFIG = {
        'min_change_threshold': 5.0,  # Minimum % change to trigger insights
        'top_n_drugs': 5,  # Number of top drugs to analyze
        'forecast_accuracy_threshold': 70.0,  # Minimum accuracy % for good forecasts
        'seasonal_strength_threshold': 0.3  # Minimum seasonal strength to detect patterns
    }
    
    # Export settings
    EXPORT_FORMATS = ['csv', 'xlsx', 'json', 'pdf']
    CHART_EXPORT_FORMATS = ['png', 'svg', 'html']
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', 'app.log')
    
    # Performance settings
    MAX_DATA_ROWS = 100000  # Maximum rows to process
    CACHE_TIMEOUT = 3600  # 1 hour cache timeout
    
    # Security settings
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
    SANITIZE_FILENAMES = True
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        
        # Set up logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            file_handler = RotatingFileHandler(
                Config.LOG_FILE, 
                maxBytes=10240000, 
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('Healthcare Sales Analytics startup')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use environment variables in production
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # Enhanced security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    WTF_CSRF_ENABLED = False


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}