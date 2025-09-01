"""
Database configuration for MySQL
"""

import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Get the directory containing this file
    config_dir = Path(__file__).parent
    # Load .env from the document_portal root directory
    env_path = config_dir.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    # If python-dotenv is not installed, try to load manually
    pass

class DatabaseConfig:
    """Database configuration settings"""
    
    # MySQL Connection Settings
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "document_portal")
    
    # Connection Pool Settings
    MYSQL_POOL_SIZE = int(os.getenv("MYSQL_POOL_SIZE", "10"))
    MYSQL_MAX_OVERFLOW = int(os.getenv("MYSQL_MAX_OVERFLOW", "20"))
    MYSQL_POOL_TIMEOUT = int(os.getenv("MYSQL_POOL_TIMEOUT", "30"))
    MYSQL_POOL_RECYCLE = int(os.getenv("MYSQL_POOL_RECYCLE", "3600"))
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get MySQL database URL for SQLAlchemy"""
        from urllib.parse import quote_plus
        # Properly encode the password to handle special characters
        encoded_password = quote_plus(cls.MYSQL_PASSWORD)
        return f"mysql+pymysql://{cls.MYSQL_USER}:{encoded_password}@{cls.MYSQL_HOST}:{cls.MYSQL_PORT}/{cls.MYSQL_DATABASE}"
    
    @classmethod
    def get_connection_params(cls) -> dict:
        """Get connection parameters for MySQL"""
        return {
            "host": cls.MYSQL_HOST,
            "port": cls.MYSQL_PORT,
            "user": cls.MYSQL_USER,
            "password": cls.MYSQL_PASSWORD,
            "database": cls.MYSQL_DATABASE,
            "charset": "utf8mb4",
            "autocommit": False
        }

# Environment variables you can set in .env file:
"""
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=document_user
MYSQL_PASSWORD=your_secure_password
MYSQL_DATABASE=document_portal
MYSQL_POOL_SIZE=10
MYSQL_MAX_OVERFLOW=20
MYSQL_POOL_TIMEOUT=30
MYSQL_POOL_RECYCLE=3600
"""
