"""
Semantic embedding service for text analysis and similarity detection.
Handles transformer model loading, text embedding generation, and storage.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from pathlib import Path
import hashlib

# Mock imports for development/testing when ML libraries aren't available
try:
    from sentence_transformers import SentenceTransformer
    import torch
    from transformers import AutoTokenizer, AutoModel
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    # Mock classes for development
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                return np.random.rand(384)
            return np.random.rand(len(texts), 384)
    
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class torch:
        @staticmethod
        def device(device_name):
            return device_name

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating and managing text embeddings using transformer models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "./models"):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize models
        self.sentence_model = None
        self.tokenizer = None
        self.bert_model = None
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Embedding storage
        self.embeddings_cache = {}
        self.embedding_storage_path = self.cache_dir / "embeddings_cache.pkl"
        
        # Load cached embeddings if they exist
        self._load_embedding_cache()
    
    async def initialize_models(self):
        """Initialize transformer models asynchronously."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            
            if not ML_LIBRARIES_AVAILABLE:
                logger.warning("ML libraries not available, using mock models")
                self.sentence_model = SentenceTransformer()
                self.tokenizer = AutoTokenizer()
                self.bert_model = AutoModel()
                return
            
            # Load sentence transformer model
            loop = asyncio.get_event_loop()
            self.sentence_model = await loop.run_in_executor(
                self.executor, 
                self._load_sentence_transformer
            )
            
            # Load BERT model for additional embeddings if needed
            bert_model_name = "bert-base-uncased"
            self.tokenizer, self.bert_model = await loop.run_in_executor(
                self.executor,
                self._load_bert_model,
                bert_model_name
            )
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_sentence_transformer(self) -> SentenceTransformer:
        """Load sentence transformer model."""
        return SentenceTransformer(self.model_name, cache_folder=str(self.cache_dir))
    
    def _load_bert_model(self, model_name: str) -> Tuple[AutoTokenizer, AutoModel]:
        """Load BERT tokenizer and model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = AutoModel.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        return tokenizer, model
    
    async def generate_embeddings(self, texts: List[str], use_sentence_bert: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            use_sentence_bert: Whether to use sentence-BERT (True) or regular BERT (False)
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        if not ML_LIBRARIES_AVAILABLE:
            # Return mock embeddings for development/testing
            logger.warning("ML libraries not available, returning mock embeddings")
            return np.random.rand(len(texts), 384)
        
        if not self.sentence_model:
            await self.initialize_models()
        
        try:
            if use_sentence_bert:
                return await self._generate_sentence_embeddings(texts)
            else:
                return await self._generate_bert_embeddings(texts)
                
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def _generate_sentence_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-BERT."""
        loop = asyncio.get_event_loop()
        
        # Fix: Use lambda to properly pass kwargs
        def encode_texts():
            return self.sentence_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
        
        embeddings = await loop.run_in_executor(
            self.executor,
            encode_texts
        )
        return embeddings
    
    async def _generate_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using regular BERT with mean pooling."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._bert_encode_batch,
            texts
        )
        return embeddings
    
    def _bert_encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode texts using BERT with mean pooling."""
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Mean pooling
                token_embeddings = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']
                
                # Apply attention mask and compute mean
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_embedding = sum_embeddings / sum_mask
                
                embeddings.append(mean_embedding.squeeze().numpy())
        
        return np.array(embeddings)
    
    async def process_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Process texts in batches for memory efficiency.
        
        Args:
            texts: List of texts to process
            batch_size: Size of each batch
            
        Returns:
            Combined embeddings array
        """
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.generate_embeddings(batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])
    
    def store_embeddings(self, text_ids: List[str], embeddings: np.ndarray):
        """
        Store embeddings with associated text IDs.
        
        Args:
            text_ids: List of unique identifiers for texts
            embeddings: Corresponding embeddings array
        """
        if len(text_ids) != len(embeddings):
            raise ValueError("Number of text IDs must match number of embeddings")
        
        for text_id, embedding in zip(text_ids, embeddings):
            self.embeddings_cache[text_id] = embedding
        
        # Save to disk
        self._save_embedding_cache()
    
    def retrieve_embeddings(self, text_ids: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Retrieve stored embeddings by text IDs.
        
        Args:
            text_ids: List of text IDs to retrieve
            
        Returns:
            Dictionary mapping text IDs to embeddings (None if not found)
        """
        return {text_id: self.embeddings_cache.get(text_id) for text_id in text_ids}
    
    def _load_embedding_cache(self):
        """Load embedding cache from disk."""
        try:
            if self.embedding_storage_path.exists():
                with open(self.embedding_storage_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embeddings_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Could not load embedding cache: {e}")
            self.embeddings_cache = {}
    
    def _save_embedding_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.embedding_storage_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            logger.error(f"Could not save embedding cache: {e}")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy array embedding with shape (embedding_dim,)
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the current model."""
        if ML_LIBRARIES_AVAILABLE and self.sentence_model and hasattr(self.sentence_model, 'get_sentence_embedding_dimension'):
            return self.sentence_model.get_sentence_embedding_dimension()
        return 384  # Default for all-MiniLM-L6-v2
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embeddings_cache.clear()
        if self.embedding_storage_path.exists():
            self.embedding_storage_path.unlink()
        logger.info("Embedding cache cleared")


# Global instance
embedding_service = EmbeddingService()