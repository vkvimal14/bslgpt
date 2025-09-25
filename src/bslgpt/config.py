"""Configuration management for BSL GPT application."""

import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Config:
    """Base configuration class."""
    
    # Flask Configuration
    SECRET_KEY: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"))
    FLASK_ENV: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "development"))
    FLASK_DEBUG: bool = field(default_factory=lambda: os.getenv("FLASK_DEBUG", "False").lower() == "true")
    
    # AI Configuration
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp"))
    MAX_OUTPUT_TOKENS: int = field(default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "8192")))
    TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    TOP_P: float = field(default_factory=lambda: float(os.getenv("TOP_P", "0.95")))
    TOP_K: int = field(default_factory=lambda: int(os.getenv("TOP_K", "40")))
    
    # File and Storage Configuration
    PDF_FOLDER: str = field(default_factory=lambda: os.getenv("PDF_FOLDER", "./pdf"))
    DATABASE_URL: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///bslgpt.db"))
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    
    # Application Settings
    MAX_FILE_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_FILE_SIZE", "10485760")))  # 10MB
    ALLOWED_EXTENSIONS: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_EXTENSIONS", "pdf,docx,txt").split(","))
    CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))  # 1 hour
    
    # Performance Settings
    MAX_WORKERS: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    TIMEOUT: int = field(default_factory=lambda: int(os.getenv("TIMEOUT", "30")))
    
    # Logging Configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.GEMINI_API_KEY:
            print("⚠️  Warning: GEMINI_API_KEY not set. Using test key for development.")
            self.GEMINI_API_KEY = "test-key"
        
        if not os.path.exists(self.PDF_FOLDER):
            os.makedirs(self.PDF_FOLDER, exist_ok=True)


@dataclass
class DevelopmentConfig(Config):
    """Development configuration."""
    FLASK_DEBUG: bool = field(default_factory=lambda: True)
    LOG_LEVEL: str = field(default_factory=lambda: "DEBUG")


@dataclass
class ProductionConfig(Config):
    """Production configuration."""
    FLASK_DEBUG: bool = field(default_factory=lambda: False)
    LOG_LEVEL: str = field(default_factory=lambda: "INFO")


@dataclass
class TestingConfig(Config):
    """Testing configuration."""
    TESTING: bool = field(default_factory=lambda: True)
    DATABASE_URL: str = field(default_factory=lambda: "sqlite:///:memory:")
    GEMINI_API_KEY: str = field(default_factory=lambda: "test-api-key")


def get_config() -> Config:
    """Get configuration based on environment."""
    env = os.getenv("FLASK_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()