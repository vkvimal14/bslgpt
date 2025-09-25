"""Configuration management for BSL GPT application."""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Base configuration class."""
    
    # Flask Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    FLASK_ENV: str = os.getenv("FLASK_ENV", "development")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    
    # AI Configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_P: float = float(os.getenv("TOP_P", "0.95"))
    TOP_K: int = int(os.getenv("TOP_K", "40"))
    
    # File and Storage Configuration
    PDF_FOLDER: str = os.getenv("PDF_FOLDER", "./pdf")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///bslgpt.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Application Settings
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    ALLOWED_EXTENSIONS: list = os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt").split(",")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
    
    # Performance Settings
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))
    TIMEOUT: int = int(os.getenv("TIMEOUT", "30"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        if not os.path.exists(self.PDF_FOLDER):
            os.makedirs(self.PDF_FOLDER, exist_ok=True)


@dataclass
class DevelopmentConfig(Config):
    """Development configuration."""
    FLASK_DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"


@dataclass
class ProductionConfig(Config):
    """Production configuration."""
    FLASK_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"


@dataclass
class TestingConfig(Config):
    """Testing configuration."""
    TESTING: bool = True
    DATABASE_URL: str = "sqlite:///:memory:"
    GEMINI_API_KEY: str = "test-api-key"


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("FLASK_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()