"""
Application configuration settings.
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    DEBUG: bool = False
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://trueauthor_user:trueauthor_password@localhost:5432/trueauthor_prod"
    )
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # JWT
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "jwt-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = "uploads"
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".docx", ".txt"]
    
    # ML Models
    MODEL_CACHE_DIR: str = "models"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Blockchain
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon-rpc.com")
    CONTRACT_ADDRESS: Optional[str] = os.getenv("CONTRACT_ADDRESS")
    PRIVATE_KEY: Optional[str] = os.getenv("PRIVATE_KEY")
    PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS: Optional[str] = os.getenv("PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS")
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
    
    # IPFS
    IPFS_API_URL: str = os.getenv("IPFS_API_URL", "http://localhost:5001")
    IPFS_GATEWAY_URL: str = os.getenv("IPFS_GATEWAY_URL", "https://ipfs.io/ipfs/")
    IPFS_PIN_ENABLED: bool = os.getenv("IPFS_PIN_ENABLED", "false").lower() == "true"
    IPFS_API_KEY: Optional[str] = os.getenv("IPFS_API_KEY")
    IPFS_API_SECRET: Optional[str] = os.getenv("IPFS_API_SECRET")
    IPFS_ENCRYPTION_PASSWORD: str = os.getenv("IPFS_ENCRYPTION_PASSWORD", "default-encryption-key-change-in-production")
    IPFS_REDUNDANCY_NODES: Optional[List[str]] = None
    
    # Security Configuration
    ENCRYPTION_MASTER_KEY: str = os.getenv("ENCRYPTION_MASTER_KEY", "change-this-master-key-in-production")
    ENABLE_DATA_ENCRYPTION: bool = os.getenv("ENABLE_DATA_ENCRYPTION", "true").lower() == "true"
    ENABLE_INTRUSION_DETECTION: bool = os.getenv("ENABLE_INTRUSION_DETECTION", "true").lower() == "true"
    SECURITY_MONITORING_ENABLED: bool = os.getenv("SECURITY_MONITORING_ENABLED", "true").lower() == "true"
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100"))
    RATE_LIMIT_BURST_SIZE: int = int(os.getenv("RATE_LIMIT_BURST_SIZE", "20"))
    
    # Session Security
    SESSION_TIMEOUT_MINUTES: int = int(os.getenv("SESSION_TIMEOUT_MINUTES", "60"))
    FORCE_HTTPS: bool = os.getenv("FORCE_HTTPS", "false").lower() == "true"
    SECURE_COOKIES: bool = os.getenv("SECURE_COOKIES", "false").lower() == "true"
    
    # Compliance
    GDPR_COMPLIANCE_ENABLED: bool = os.getenv("GDPR_COMPLIANCE_ENABLED", "true").lower() == "true"
    FERPA_COMPLIANCE_ENABLED: bool = os.getenv("FERPA_COMPLIANCE_ENABLED", "true").lower() == "true"
    DATA_RETENTION_ENABLED: bool = os.getenv("DATA_RETENTION_ENABLED", "true").lower() == "true"
    
    # Audit Logging
    AUDIT_LOG_RETENTION_DAYS: int = int(os.getenv("AUDIT_LOG_RETENTION_DAYS", "365"))
    AUDIT_LOG_ENCRYPTION: bool = os.getenv("AUDIT_LOG_ENCRYPTION", "true").lower() == "true"
    
    # Email Configuration
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "localhost")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME: str = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS: bool = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "noreply@stylos.edu")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    class Config:
        env_file = ".env"


settings = Settings()