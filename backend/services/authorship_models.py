"""
Authorship verification models using Siamese networks and AI detection.

This module implements deep learning models for:
1. Siamese network for authorship comparison
2. AI-generated content detection
3. Confidence interval calculation and uncertainty quantification
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from pathlib import Path
import json
from dataclasses import dataclass

# Mock imports for development/testing when ML libraries aren't available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import train_test_split
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    # Mock classes for development
    class nn:
        class Module:
            def __init__(self):
                pass
            def forward(self, x):
                return x
            def train(self):
                pass
            def eval(self):
                pass
            def parameters(self):
                return []
        class Linear:
            def __init__(self, *args, **kwargs):
                pass
        class Dropout:
            def __init__(self, *args, **kwargs):
                pass
        class ReLU:
            def __init__(self, *args, **kwargs):
                pass
    
    class torch:
        @staticmethod
        def tensor(data):
            return np.array(data)
        @staticmethod
        def no_grad():
            return None
        @staticmethod
        def save(model, path):
            pass
        @staticmethod
        def load(path):
            return None

logger = logging.getLogger(__name__)

@dataclass
class AuthorshipResult:
    """Result of authorship verification."""
    similarity_score: float
    confidence_interval: Tuple[float, float]
    is_authentic: bool
    uncertainty: float
    feature_importance: Dict[str, float]

@dataclass
class AIDetectionResult:
    """Result of AI-generated content detection."""
    ai_probability: float
    human_probability: float
    confidence: float
    detection_method: str
    explanation: Dict[str, Any]

class SiameseNetwork(nn.Module if ML_LIBRARIES_AVAILABLE else object):
    """
    Siamese neural network for authorship comparison.
    
    This network learns to compare two text embeddings and determine
    if they are from the same author.
    """
    
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, dropout_rate: float = 0.3):
        """
        Initialize the Siamese network.
        
        Args:
            input_dim: Dimension of input embeddings (384 for all-MiniLM-L6-v2)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
        """
        if ML_LIBRARIES_AVAILABLE:
            super(SiameseNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if not ML_LIBRARIES_AVAILABLE:
            logger.warning("ML libraries not available, using mock Siamese network")
            return
        
        # Shared feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Comparison layers
        self.comparison_layers = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward_one(self, x):
        """Forward pass for one input."""
        if not ML_LIBRARIES_AVAILABLE:
            return np.random.rand(self.hidden_dim // 2)
        return self.feature_extractor(x)
    
    def forward(self, input1, input2):
        """
        Forward pass for Siamese network.
        
        Args:
            input1: First text embedding
            input2: Second text embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        if not ML_LIBRARIES_AVAILABLE:
            return np.random.rand(1)
        
        # Extract features for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        
        # Calculate absolute difference
        diff = torch.abs(output1 - output2)
        
        # Pass through comparison layers
        similarity = self.comparison_layers(diff)
        
        return similarity

class AuthorshipDataset(Dataset if ML_LIBRARIES_AVAILABLE else object):
    """Dataset for training Siamese network with positive/negative pairs."""
    
    def __init__(self, embeddings: List[np.ndarray], labels: List[int], author_ids: List[str]):
        """
        Initialize the dataset.
        
        Args:
            embeddings: List of text embeddings
            labels: List of labels (1 for same author, 0 for different)
            author_ids: List of author identifiers
        """
        self.embeddings = embeddings
        self.labels = labels
        self.author_ids = author_ids
        
        # Generate pairs
        self.pairs, self.pair_labels = self._generate_pairs()
    
    def _generate_pairs(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Generate positive and negative pairs for training."""
        pairs = []
        pair_labels = []
        
        # Create author groups
        author_groups = {}
        for i, author_id in enumerate(self.author_ids):
            if author_id not in author_groups:
                author_groups[author_id] = []
            author_groups[author_id].append(i)
        
        # Generate positive pairs (same author)
        for author_id, indices in author_groups.items():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        pairs.append((indices[i], indices[j]))
                        pair_labels.append(1)
        
        # Generate negative pairs (different authors)
        authors = list(author_groups.keys())
        for i in range(len(authors)):
            for j in range(i + 1, len(authors)):
                author1_indices = author_groups[authors[i]]
                author2_indices = author_groups[authors[j]]
                
                # Sample a few negative pairs to balance the dataset
                for idx1 in author1_indices[:2]:  # Limit to avoid too many negatives
                    for idx2 in author2_indices[:2]:
                        pairs.append((idx1, idx2))
                        pair_labels.append(0)
        
        return pairs, pair_labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if not ML_LIBRARIES_AVAILABLE:
            return (np.random.rand(768), np.random.rand(768)), 0
        
        pair_idx1, pair_idx2 = self.pairs[idx]
        embedding1 = torch.tensor(self.embeddings[pair_idx1], dtype=torch.float32)
        embedding2 = torch.tensor(self.embeddings[pair_idx2], dtype=torch.float32)
        label = torch.tensor(self.pair_labels[idx], dtype=torch.float32)
        
        return (embedding1, embedding2), label

class AIDetectionClassifier:
    """
    Ensemble classifier for AI-generated content detection.
    
    Combines multiple approaches for robust AI detection.
    """
    
    def __init__(self, model_dir: str = "./models/ai_detection"):
        """
        Initialize the AI detection classifier.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ensemble components
        self.stylometric_classifier = None
        self.transformer_classifier = None
        self.ensemble_classifier = None
        
        # Feature extractors
        self.tokenizer = None
        self.transformer_model = None
        
    async def initialize_models(self):
        """Initialize the AI detection models."""
        try:
            if not ML_LIBRARIES_AVAILABLE:
                logger.warning("ML libraries not available, using mock AI detection")
                return
            
            # Load transformer model for feature extraction
            model_name = "distilbert-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            
            # Initialize classifiers
            self.stylometric_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            
            self.transformer_classifier = LogisticRegression(
                random_state=42,
                class_weight='balanced'
            )
            
            # Try to load pre-trained models
            await self._load_pretrained_models()
            
            logger.info("AI detection models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing AI detection models: {e}")
            raise
    
    async def _load_pretrained_models(self):
        """Load pre-trained models if available."""
        try:
            stylometric_path = self.model_dir / "stylometric_classifier.pkl"
            transformer_path = self.model_dir / "transformer_classifier.pkl"
            
            if stylometric_path.exists():
                with open(stylometric_path, 'rb') as f:
                    self.stylometric_classifier = pickle.load(f)
                logger.info("Loaded pre-trained stylometric classifier")
            
            if transformer_path.exists():
                with open(transformer_path, 'rb') as f:
                    self.transformer_classifier = pickle.load(f)
                logger.info("Loaded pre-trained transformer classifier")
                
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")
    
    def extract_transformer_features(self, texts: List[str]) -> np.ndarray:
        """Extract features using transformer model."""
        if not ML_LIBRARIES_AVAILABLE or not self.tokenizer:
            # Return mock features for development
            return np.random.rand(len(texts), 768)
        
        features = []
        
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
                outputs = self.transformer_model(**inputs)
                # Use CLS token embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                features.append(cls_embedding.numpy())
        
        return np.array(features)
    
    async def train(self, texts: List[str], labels: List[int], stylometric_features: List[Dict[str, float]]):
        """
        Train the AI detection ensemble.
        
        Args:
            texts: List of text samples
            labels: List of labels (0 for human, 1 for AI)
            stylometric_features: List of stylometric feature dictionaries
        """
        if not ML_LIBRARIES_AVAILABLE:
            logger.warning("ML libraries not available, skipping training")
            return
        
        try:
            # Prepare stylometric features
            stylometric_matrix = self._prepare_stylometric_matrix(stylometric_features)
            
            # Extract transformer features
            transformer_features = self.extract_transformer_features(texts)
            
            # Split data
            (stylo_train, stylo_test, trans_train, trans_test, 
             y_train, y_test) = train_test_split(
                stylometric_matrix, transformer_features, labels,
                test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train individual classifiers
            self.stylometric_classifier.fit(stylo_train, y_train)
            self.transformer_classifier.fit(trans_train, y_train)
            
            # Create ensemble
            self.ensemble_classifier = VotingClassifier(
                estimators=[
                    ('stylometric', self.stylometric_classifier),
                    ('transformer', self.transformer_classifier)
                ],
                voting='soft'
            )
            
            # Train ensemble on combined features
            combined_train = np.hstack([stylo_train, trans_train])
            combined_test = np.hstack([stylo_test, trans_test])
            
            # For ensemble, we need to retrain on combined features
            ensemble_stylo = RandomForestClassifier(n_estimators=100, random_state=42)
            ensemble_trans = LogisticRegression(random_state=42)
            
            ensemble_stylo.fit(stylo_train, y_train)
            ensemble_trans.fit(trans_train, y_train)
            
            # Evaluate models
            stylo_pred = self.stylometric_classifier.predict(stylo_test)
            trans_pred = self.transformer_classifier.predict(trans_test)
            
            stylo_acc = accuracy_score(y_test, stylo_pred)
            trans_acc = accuracy_score(y_test, trans_pred)
            
            logger.info(f"Stylometric classifier accuracy: {stylo_acc:.3f}")
            logger.info(f"Transformer classifier accuracy: {trans_acc:.3f}")
            
            # Save models
            await self._save_models()
            
        except Exception as e:
            logger.error(f"Error training AI detection models: {e}")
            raise
    
    def _prepare_stylometric_matrix(self, stylometric_features: List[Dict[str, float]]) -> np.ndarray:
        """Convert stylometric features to matrix format."""
        if not stylometric_features:
            return np.array([])
        
        # Get all feature names
        all_features = set()
        for features in stylometric_features:
            all_features.update(features.keys())
        
        feature_names = sorted(list(all_features))
        
        # Create matrix
        matrix = []
        for features in stylometric_features:
            row = [features.get(name, 0.0) for name in feature_names]
            matrix.append(row)
        
        return np.array(matrix)
    
    async def detect_ai_content(self, text: str, stylometric_features: Dict[str, float]) -> AIDetectionResult:
        """
        Detect if content is AI-generated.
        
        Args:
            text: Text to analyze
            stylometric_features: Stylometric features of the text
            
        Returns:
            AI detection result with probabilities and explanation
        """
        try:
            if not ML_LIBRARIES_AVAILABLE or not self.stylometric_classifier:
                # Return mock result for development
                ai_prob = np.random.rand()
                return AIDetectionResult(
                    ai_probability=ai_prob,
                    human_probability=1.0 - ai_prob,
                    confidence=0.7,
                    detection_method="mock",
                    explanation={"note": "Mock detection for development"}
                )
            
            # Prepare features
            stylo_features = np.array([[stylometric_features.get(key, 0.0) 
                                     for key in sorted(stylometric_features.keys())]])
            trans_features = self.extract_transformer_features([text])
            
            # Get predictions from individual classifiers
            stylo_proba = self.stylometric_classifier.predict_proba(stylo_features)[0]
            trans_proba = self.transformer_classifier.predict_proba(trans_features)[0]
            
            # Ensemble prediction (weighted average)
            ensemble_proba = (stylo_proba + trans_proba) / 2
            
            ai_probability = float(ensemble_proba[1])  # Probability of AI-generated
            human_probability = float(ensemble_proba[0])  # Probability of human-written
            
            # Calculate confidence as the difference between probabilities
            confidence = abs(ai_probability - human_probability)
            
            # Feature importance for explanation
            feature_importance = {}
            if hasattr(self.stylometric_classifier, 'feature_importances_'):
                feature_names = sorted(stylometric_features.keys())
                importances = self.stylometric_classifier.feature_importances_
                feature_importance = dict(zip(feature_names, importances))
            
            explanation = {
                "stylometric_prediction": float(stylo_proba[1]),
                "transformer_prediction": float(trans_proba[1]),
                "feature_importance": feature_importance,
                "method": "ensemble_voting"
            }
            
            return AIDetectionResult(
                ai_probability=ai_probability,
                human_probability=human_probability,
                confidence=confidence,
                detection_method="ensemble",
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error in AI detection: {e}")
            # Return neutral result on error
            return AIDetectionResult(
                ai_probability=0.5,
                human_probability=0.5,
                confidence=0.0,
                detection_method="error",
                explanation={"error": str(e)}
            )
    
    async def _save_models(self):
        """Save trained models to disk."""
        try:
            if self.stylometric_classifier:
                with open(self.model_dir / "stylometric_classifier.pkl", 'wb') as f:
                    pickle.dump(self.stylometric_classifier, f)
            
            if self.transformer_classifier:
                with open(self.model_dir / "transformer_classifier.pkl", 'wb') as f:
                    pickle.dump(self.transformer_classifier, f)
                    
            logger.info("AI detection models saved")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

class AuthorshipVerificationService:
    """
    Main service for authorship verification using Siamese networks.
    """
    
    def __init__(self, model_dir: str = "./models/authorship"):
        """
        Initialize the authorship verification service.
        
        Args:
            model_dir: Directory to store trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.siamese_model = None
        self.ai_detector = AIDetectionClassifier()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize all models and services."""
        try:
            # Initialize Siamese network
            self.siamese_model = SiameseNetwork()
            
            # Try to load pre-trained model
            model_path = self.model_dir / "siamese_model.pth"
            if model_path.exists() and ML_LIBRARIES_AVAILABLE:
                self.siamese_model.load_state_dict(torch.load(model_path, weights_only=True))
                logger.info("Loaded pre-trained Siamese model")
            
            # Initialize AI detector
            await self.ai_detector.initialize_models()
            
            logger.info("Authorship verification service initialized")
            
        except Exception as e:
            logger.error(f"Error initializing authorship verification service: {e}")
            raise
    
    async def train_siamese_model(self, embeddings: List[np.ndarray], author_ids: List[str], 
                                epochs: int = 50, batch_size: int = 32):
        """
        Train the Siamese network for authorship comparison.
        
        Args:
            embeddings: List of text embeddings
            author_ids: Corresponding author identifiers
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not ML_LIBRARIES_AVAILABLE:
            logger.warning("ML libraries not available, skipping Siamese training")
            return
        
        try:
            # Create dataset
            labels = [1] * len(embeddings)  # Placeholder labels
            dataset = AuthorshipDataset(embeddings, labels, author_ids)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training setup
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(self.siamese_model.parameters(), lr=0.001)
            
            self.siamese_model.train()
            
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0
                
                for batch_idx, ((input1, input2), labels) in enumerate(dataloader):
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.siamese_model(input1, input2)
                    loss = criterion(outputs.squeeze(), labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save trained model
            torch.save(self.siamese_model.state_dict(), self.model_dir / "siamese_model.pth")
            logger.info("Siamese model training completed and saved")
            
        except Exception as e:
            logger.error(f"Error training Siamese model: {e}")
            raise
    
    async def verify_authorship(self, candidate_embedding: np.ndarray, 
                              reference_embeddings: List[np.ndarray],
                              confidence_level: float = 0.95) -> AuthorshipResult:
        """
        Verify authorship by comparing candidate text with reference samples.
        
        Args:
            candidate_embedding: Embedding of text to verify
            reference_embeddings: List of reference embeddings from known author
            confidence_level: Confidence level for interval calculation
            
        Returns:
            Authorship verification result with confidence intervals
        """
        try:
            if not reference_embeddings:
                return AuthorshipResult(
                    similarity_score=0.0,
                    confidence_interval=(0.0, 0.0),
                    is_authentic=False,
                    uncertainty=1.0,
                    feature_importance={}
                )
            
            # Calculate similarities with all reference samples
            similarities = []
            
            if ML_LIBRARIES_AVAILABLE and self.siamese_model:
                self.siamese_model.eval()
                with torch.no_grad():
                    candidate_tensor = torch.tensor(candidate_embedding, dtype=torch.float32)
                    
                    for ref_embedding in reference_embeddings:
                        ref_tensor = torch.tensor(ref_embedding, dtype=torch.float32)
                        similarity = self.siamese_model(candidate_tensor, ref_tensor)
                        similarities.append(float(similarity.item()))
            else:
                # Fallback to cosine similarity for development
                from sklearn.metrics.pairwise import cosine_similarity
                for ref_embedding in reference_embeddings:
                    sim = cosine_similarity([candidate_embedding], [ref_embedding])[0][0]
                    similarities.append(float(sim))
            
            # Calculate statistics
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Calculate confidence interval
            n = len(similarities)
            if n > 1:
                # Use t-distribution for small samples
                from scipy import stats
                t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
                margin_error = t_value * (std_similarity / np.sqrt(n))
            else:
                margin_error = std_similarity
            
            confidence_interval = (
                max(0.0, mean_similarity - margin_error),
                min(1.0, mean_similarity + margin_error)
            )
            
            # Determine authenticity (threshold can be adjusted)
            authenticity_threshold = 0.7
            is_authentic = mean_similarity >= authenticity_threshold
            
            # Calculate uncertainty
            uncertainty = std_similarity / (mean_similarity + 1e-10)
            
            # Feature importance (simplified)
            feature_importance = {
                "mean_similarity": mean_similarity,
                "std_similarity": std_similarity,
                "sample_count": n
            }
            
            return AuthorshipResult(
                similarity_score=mean_similarity,
                confidence_interval=confidence_interval,
                is_authentic=is_authentic,
                uncertainty=uncertainty,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error in authorship verification: {e}")
            return AuthorshipResult(
                similarity_score=0.0,
                confidence_interval=(0.0, 0.0),
                is_authentic=False,
                uncertainty=1.0,
                feature_importance={"error": str(e)}
            )

# Global instances
authorship_service = AuthorshipVerificationService()
ai_detection_service = AIDetectionClassifier()