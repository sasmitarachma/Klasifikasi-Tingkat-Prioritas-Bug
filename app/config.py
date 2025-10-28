import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_uploads')
    ALLOWED_EXTENSIONS = {'csv', 'txt'}
    
    # ML Model settings
    ML_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_model')
    MODEL_CACHE_TIMEOUT = 3600  # 1 hour cache for loaded models
    
    # Export settings
    EXPORT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_exports')
    EXPORT_FILE_LIFETIME = timedelta(hours=1)  # Auto-delete exported files after 1 hour
    
    # GitHub API settings
    GITHUB_API_TIMEOUT = 30  # seconds
    GITHUB_MAX_ISSUES = 100  # maximum issues to fetch per request
    GITHUB_RATE_LIMIT_DELAY = 1  # seconds between API calls
    
    # Application settings
    PAGINATION_PER_PAGE = 20
    MAX_CONTENT_LENGTH_MB = 16
    
    # Security settings
    WTF_CSRF_ENABLED = False  # Disabled for simplicity, enable in production
    WTF_CSRF_TIME_LIMIT = None
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        # Create necessary directories
        dirs_to_create = [
            Config.UPLOAD_FOLDER,
            Config.EXPORT_FOLDER,
            Config.ML_MODEL_PATH
        ]
        
        for directory in dirs_to_create:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except OSError as e:
                    app.logger.warning(f"Could not create directory {directory}: {e}")

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Relaxed file size limits for development
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB for development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Stricter settings for production
    LOG_LEVEL = 'WARNING'
    WTF_CSRF_ENABLED = True
    
    # Production security headers
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=1)
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Production-specific initialization
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug and not app.testing:
            # Set up file logging
            if not os.path.exists('logs'):
                os.mkdir('logs')
            file_handler = RotatingFileHandler('logs/bug_priority_app.log',
                                               maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            app.logger.setLevel(logging.INFO)
            app.logger.info('Bug Priority Classification startup')

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    
    # Smaller limits for testing
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1MB for testing
    GITHUB_MAX_ISSUES = 10  # Limit for testing

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Helper function to get config
def get_config(config_name=None):
    """Get configuration class based on environment"""
    if config_name is None:
        config_name = os.environ.get('FLASK_CONFIG', 'default')
    return config.get(config_name, DevelopmentConfig)