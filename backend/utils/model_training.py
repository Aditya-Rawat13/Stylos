"""
Utility functions for training and managing authorship verification models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import json
from pathlib import Path
from datetime import datetime

from ..services.authorship_integration import authorship_manager
from ..services.embedding_service import embedding_service
from ..services.stylometric_analyzer import StylometricAnalyzer

logger = logging.getLogger(__name__)

class ModelTrainingUtils:
    """Utilities for training authorship verification models."""
    
    def __init__(self):
        """Initialize training utilities."""
        self.stylometric_analyzer = StylometricAnalyzer()
    
    async def prepare_authorship_training_data(self, texts_by_author: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Prepare training data for authorship verification models.
        
        Args:
            texts_by_author: Dictionary mapping author IDs to their texts
            
        Returns:
            Prepared training data
        """
        try:
            logger.info("Preparing authorship training data...")
            
            all_texts = []
            author_ids = []
            embeddings = []
            
            # Process texts for each author
            for author_id, texts in texts_by_author.items():
                for text in texts:
                    # Generate embedding
                    embedding = await embedding_service.generate_embedding(text)
                    
                    all_texts.append(text)
                    author_ids.append(author_id)
                    embeddings.append(embedding)
            
            training_data = {
                'texts': all_texts,
                'author_ids': author_ids,
                'embeddings': embeddings
            }
            
            logger.info(f"Prepared training data for {len(set(author_ids))} authors with {len(all_texts)} texts")
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing authorship training data: {e}")
            raise
    
    async def prepare_ai_detection_training_data(self, human_texts: List[str], 
                                               ai_texts: List[str]) -> Dict[str, Any]:
        """
        Prepare training data for AI detection models.
        
        Args:
            human_texts: List of human-written texts
            ai_texts: List of AI-generated texts
            
        Returns:
            Prepared training data
        """
        try:
            logger.info("Preparing AI detection training data...")
            
            all_texts = human_texts + ai_texts
            labels = [0] * len(human_texts) + [1] * len(ai_texts)  # 0 = human, 1 = AI
            
            # Extract stylometric features for all texts
            stylometric_features = []
            for text in all_texts:
                features = self.stylometric_analyzer.extract_features(text)
                stylometric_features.append(features)
            
            training_data = {
                'texts': all_texts,
                'labels': labels,
                'stylometric_features': stylometric_features
            }
            
            logger.info(f"Prepared AI detection training data: {len(human_texts)} human, {len(ai_texts)} AI texts")
            return training_data
            
        except Exception as e:
            logger.error(f"Error preparing AI detection training data: {e}")
            raise
    
    def load_training_data_from_csv(self, csv_path: str, 
                                  text_column: str = 'text',
                                  author_column: str = 'author',
                                  label_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Load training data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            text_column: Name of text column
            author_column: Name of author column
            label_column: Name of label column (for AI detection)
            
        Returns:
            Loaded training data
        """
        try:
            df = pd.read_csv(csv_path)
            
            if label_column and label_column in df.columns:
                # AI detection data
                human_texts = df[df[label_column] == 0][text_column].tolist()
                ai_texts = df[df[label_column] == 1][text_column].tolist()
                
                return {
                    'type': 'ai_detection',
                    'human_texts': human_texts,
                    'ai_texts': ai_texts
                }
            else:
                # Authorship data
                texts_by_author = {}
                for _, row in df.iterrows():
                    author = row[author_column]
                    text = row[text_column]
                    
                    if author not in texts_by_author:
                        texts_by_author[author] = []
                    texts_by_author[author].append(text)
                
                return {
                    'type': 'authorship',
                    'texts_by_author': texts_by_author
                }
                
        except Exception as e:
            logger.error(f"Error loading training data from CSV: {e}")
            raise
    
    async def train_models_from_data(self, training_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Train models using prepared training data.
        
        Args:
            training_data: Prepared training data
            
        Returns:
            Training results
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare data for different model types
            model_training_data = {}
            
            if 'authorship' in training_data:
                authorship_data = await self.prepare_authorship_training_data(
                    training_data['authorship']['texts_by_author']
                )
                model_training_data['authorship_data'] = authorship_data
            
            if 'ai_detection' in training_data:
                ai_data = await self.prepare_ai_detection_training_data(
                    training_data['ai_detection']['human_texts'],
                    training_data['ai_detection']['ai_texts']
                )
                model_training_data['ai_detection_data'] = ai_data
            
            # Train models
            results = await authorship_manager.train_models_with_data(model_training_data)
            
            logger.info("Model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
    
    def generate_synthetic_training_data(self, num_authors: int = 10, 
                                       texts_per_author: int = 5) -> Dict[str, List[str]]:
        """
        Generate synthetic training data for testing purposes.
        
        Args:
            num_authors: Number of synthetic authors
            texts_per_author: Number of texts per author
            
        Returns:
            Synthetic training data
        """
        try:
            logger.info(f"Generating synthetic training data for {num_authors} authors")
            
            texts_by_author = {}
            
            # Define different writing styles
            styles = [
                {
                    'vocab': ['analyze', 'examine', 'investigate', 'determine', 'establish'],
                    'structure': 'formal',
                    'sentence_length': 'long'
                },
                {
                    'vocab': ['think', 'believe', 'feel', 'see', 'know'],
                    'structure': 'informal',
                    'sentence_length': 'short'
                },
                {
                    'vocab': ['demonstrate', 'illustrate', 'exemplify', 'show', 'reveal'],
                    'structure': 'academic',
                    'sentence_length': 'medium'
                }
            ]
            
            for i in range(num_authors):
                author_id = f"synthetic_author_{i+1}"
                style = styles[i % len(styles)]
                
                texts = []
                for j in range(texts_per_author):
                    text = self._generate_synthetic_text(style, j)
                    texts.append(text)
                
                texts_by_author[author_id] = texts
            
            logger.info("Synthetic training data generated successfully")
            return texts_by_author
            
        except Exception as e:
            logger.error(f"Error generating synthetic training data: {e}")
            raise
    
    def _generate_synthetic_text(self, style: Dict[str, str], seed: int) -> str:
        """Generate a synthetic text with given style."""
        np.random.seed(seed)
        
        vocab = style['vocab']
        structure = style['structure']
        sentence_length = style['sentence_length']
        
        # Generate sentences based on style
        sentences = []
        num_sentences = np.random.randint(3, 8)
        
        for _ in range(num_sentences):
            if sentence_length == 'short':
                words_per_sentence = np.random.randint(5, 10)
            elif sentence_length == 'medium':
                words_per_sentence = np.random.randint(10, 20)
            else:  # long
                words_per_sentence = np.random.randint(15, 30)
            
            # Build sentence
            sentence_words = []
            for i in range(words_per_sentence):
                if i == 0:
                    # Start with style-specific vocabulary
                    word = np.random.choice(vocab)
                elif i < 3:
                    # Add common words
                    word = np.random.choice(['the', 'a', 'an', 'this', 'that'])
                else:
                    # Add filler words
                    word = np.random.choice(['data', 'results', 'method', 'approach', 'system'])
                
                sentence_words.append(word)
            
            sentence = ' '.join(sentence_words).capitalize() + '.'
            sentences.append(sentence)
        
        return ' '.join(sentences)
    
    async def evaluate_model_performance(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: Test data with ground truth
            
        Returns:
            Performance evaluation results
        """
        try:
            logger.info("Evaluating model performance...")
            
            # Benchmark models
            results = await authorship_manager.benchmark_models(test_data)
            
            # Add timestamp and metadata
            results['evaluation_timestamp'] = datetime.utcnow().isoformat()
            results['test_data_size'] = {
                'authorship': len(test_data.get('authorship_test', {}).get('texts', [])),
                'ai_detection': len(test_data.get('ai_detection_test', {}).get('texts', []))
            }
            
            logger.info("Model performance evaluation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            raise
    
    def save_training_results(self, results: Dict[str, Any], output_path: str):
        """
        Save training results to file.
        
        Args:
            results: Training results
            output_path: Path to save results
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            results['saved_timestamp'] = datetime.utcnow().isoformat()
            results['model_versions'] = {
                'authorship_verification': '1.0',
                'ai_detection': '1.0',
                'enhanced_ai_detection': '1.0'
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Training results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving training results: {e}")
            raise

# Global instance
training_utils = ModelTrainingUtils()