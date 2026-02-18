"""
Custom exceptions for Project Stylos backend services.
"""

class StylosException(Exception):
    """Base exception for Project Stylos."""
    pass

class EmbeddingError(StylosException):
    """Exception raised for embedding generation errors."""
    pass

class ModelLoadError(StylosException):
    """Exception raised when ML models fail to load."""
    pass

class SimilarityError(StylosException):
    """Exception raised for similarity calculation errors."""
    pass

class DuplicateDetectionError(StylosException):
    """Exception raised for duplicate detection errors."""
    pass

class VectorDatabaseError(StylosException):
    """Exception raised for vector database operations."""
    pass