"""
Configuration settings for authorship verification models.
"""

from typing import Dict, Any
from pathlib import Path
import os

class AuthorshipConfig:
    """Configuration class for authorship verification system."""
    
    # Model directories
    MODEL_BASE_DIR = Path("./models")
    AUTHORSHIP_MODEL_DIR = MODEL_BASE_DIR / "authorship"
    AI_DETECTION_MODEL_DIR = MODEL_BASE_DIR / "ai_detection"
    ENHANCED_AI_MODEL_DIR = MODEL_BASE_DIR / "ai_detection_enhanced"
    EMBEDDING_CACHE_DIR = MODEL_BASE_DIR / "embeddings"
    
    # Siamese Network Configuration
    SIAMESE_CONFIG = {
        'input_dim': 768,  # BERT embedding dimension
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'early_stopping_patience': 10
    }
    
    # AI Detection Configuration
    AI_DETECTION_CONFIG = {
        'ensemble_models': [
            'stylometric',
            'transformer',
            'perplexity',
            'linguistic'
        ],
        'transformer_models': [
            'distilbert-base-uncased',
            'roberta-base'
        ],
        'perplexity_model': 'gpt2',
        'confidence_threshold': 0.7,
        'ai_probability_threshold': 0.8
    }
    
    # Verification Thresholds
    VERIFICATION_THRESHOLDS = {
        'authorship_min_score': 0.7,
        'ai_detection_max_prob': 0.8,
        'duplicate_max_similarity': 0.85,
        'min_confidence': 0.6,
        'uncertainty_max': 0.5
    }
    
    # Feature Extraction Configuration
    FEATURE_CONFIG = {
        'min_text_length': 100,  # Minimum characters for analysis
        'max_text_length': 10000,  # Maximum characters for analysis
        'embedding_batch_size': 16,
        'stylometric_features': [
            'type_token_ratio',
            'mtld',
            'avg_sentence_length',
            'punctuation_patterns',
            'pos_distribution',
            'function_words',
            'readability_metrics'
        ]
    }
    
    # Training Configuration
    TRAINING_CONFIG = {
        'validation_split': 0.2,
        'test_split': 0.1,
        'cross_validation_folds': 5,
        'random_seed': 42,
        'min_samples_per_author': 3,
        'max_samples_per_author': 100
    }
    
    # Performance Monitoring
    MONITORING_CONFIG = {
        'log_predictions': True,
        'save_embeddings': True,
        'track_processing_time': True,
        'alert_thresholds': {
            'accuracy_drop': 0.1,
            'processing_time_increase': 2.0,
            'error_rate_increase': 0.05
        }
    }
    
    # API Configuration
    API_CONFIG = {
        'max_concurrent_requests': 10,
        'request_timeout_seconds': 300,
        'rate_limit_per_minute': 100,
        'max_file_size_mb': 10
    }
    
    # Explainability Configuration
    EXPLAINABILITY_CONFIG = {
        'generate_explanations': True,
        'explanation_detail_level': 'detailed',  # 'basic', 'detailed', 'comprehensive'
        'include_feature_importance': True,
        'include_confidence_intervals': True,
        'linguistic_analysis_depth': 'full'  # 'basic', 'standard', 'full'
    }
    
    @classmethod
    def get_model_path(cls, model_type: str, model_name: str) -> Path:
        """
        Get the full path for a specific model.
        
        Args:
            model_type: Type of model ('authorship', 'ai_detection', etc.)
            model_name: Name of the model file
            
        Returns:
            Full path to the model
        """
        model_dirs = {
            'authorship': cls.AUTHORSHIP_MODEL_DIR,
            'ai_detection': cls.AI_DETECTION_MODEL_DIR,
            'enhanced_ai': cls.ENHANCED_AI_MODEL_DIR,
            'embeddings': cls.EMBEDDING_CACHE_DIR
        }
        
        base_dir = model_dirs.get(model_type, cls.MODEL_BASE_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        return base_dir / model_name
    
    @classmethod
    def update_thresholds(cls, new_thresholds: Dict[str, float]):
        """
        Update verification thresholds.
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        cls.VERIFICATION_THRESHOLDS.update(new_thresholds)
    
    @classmethod
    def get_environment_config(cls) -> Dict[str, Any]:
        """
        Get configuration based on environment variables.
        
        Returns:
            Environment-specific configuration
        """
        env = os.getenv('ENVIRONMENT', 'development')
        
        if env == 'production':
            return {
                'model_cache_size': 1000,
                'enable_gpu': True,
                'log_level': 'INFO',
                'save_predictions': True,
                'enable_monitoring': True
            }
        elif env == 'testing':
            return {
                'model_cache_size': 10,
                'enable_gpu': False,
                'log_level': 'DEBUG',
                'save_predictions': False,
                'enable_monitoring': False,
                'use_mock_models': True
            }
        else:  # development
            return {
                'model_cache_size': 100,
                'enable_gpu': False,
                'log_level': 'DEBUG',
                'save_predictions': True,
                'enable_monitoring': True,
                'use_mock_models': False
            }
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """
        Validate configuration settings.
        
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        # Check model directories
        try:
            cls.MODEL_BASE_DIR.mkdir(parents=True, exist_ok=True)
            validation_results['model_directories'] = True
        except Exception:
            validation_results['model_directories'] = False
        
        # Check threshold values
        thresholds_valid = all(
            0.0 <= value <= 1.0 
            for value in cls.VERIFICATION_THRESHOLDS.values()
        )
        validation_results['thresholds'] = thresholds_valid
        
        # Check feature configuration
        features_valid = (
            cls.FEATURE_CONFIG['min_text_length'] > 0 and
            cls.FEATURE_CONFIG['max_text_length'] > cls.FEATURE_CONFIG['min_text_length']
        )
        validation_results['feature_config'] = features_valid
        
        # Check training configuration
        training_valid = (
            0.0 < cls.TRAINING_CONFIG['validation_split'] < 1.0 and
            0.0 < cls.TRAINING_CONFIG['test_split'] < 1.0 and
            cls.TRAINING_CONFIG['validation_split'] + cls.TRAINING_CONFIG['test_split'] < 1.0
        )
        validation_results['training_config'] = training_valid
        
        return validation_results

# Create global config instance
config = AuthorshipConfig()