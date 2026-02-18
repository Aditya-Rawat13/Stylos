"""
Enhanced AI-generated content detection system with multiple approaches
and explainability features for Project TrueAuthor.
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
from collections import Counter
import re

# Mock imports for development/testing when ML libraries aren't available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        RobertaTokenizer, RobertaForSequenceClassification,
        GPT2LMHeadModel, GPT2Tokenizer
    )
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.model_selection import cross_val_score
    import scipy.stats as stats
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    # Mock classes for development
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None
    
    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

logger = logging.getLogger(__name__)

@dataclass
class AIDetectionFeatures:
    """Features extracted for AI detection."""
    perplexity_score: float
    burstiness_score: float
    repetition_patterns: Dict[str, float]
    linguistic_markers: Dict[str, float]
    stylometric_anomalies: Dict[str, float]
    transformer_embeddings: np.ndarray
    attention_patterns: Optional[np.ndarray] = None

@dataclass
class DetectionExplanation:
    """Detailed explanation of AI detection decision."""
    primary_indicators: List[str]
    confidence_factors: Dict[str, float]
    linguistic_evidence: Dict[str, Any]
    model_contributions: Dict[str, float]
    risk_assessment: str
    human_readable_summary: str

class PerplexityAnalyzer:
    """Analyzer for text perplexity using language models."""
    
    def __init__(self, model_name: str = "gpt2"):
        """Initialize perplexity analyzer."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    async def initialize(self):
        """Initialize the language model for perplexity calculation."""
        if not ML_LIBRARIES_AVAILABLE:
            logger.warning("ML libraries not available, using mock perplexity analyzer")
            return
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            self.model.eval()
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Perplexity analyzer initialized with {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing perplexity analyzer: {e}")
            raise
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of text using the language model.
        
        Args:
            text: Input text
            
        Returns:
            Perplexity score (lower values indicate more predictable text)
        """
        if not ML_LIBRARIES_AVAILABLE or not self.model:
            # Return mock perplexity for development
            return np.random.uniform(10, 100)
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
            
            return float(perplexity)
            
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return 50.0  # Default moderate perplexity

class BurstinessAnalyzer:
    """Analyzer for text burstiness patterns."""
    
    def calculate_burstiness(self, text: str) -> float:
        """
        Calculate burstiness score based on word repetition patterns.
        
        AI-generated text often has lower burstiness (more uniform distribution).
        
        Args:
            text: Input text
            
        Returns:
            Burstiness score (higher values indicate more human-like patterns)
        """
        try:
            words = text.lower().split()
            if len(words) < 10:
                return 0.5  # Neutral score for short texts
            
            # Count word frequencies
            word_counts = Counter(words)
            frequencies = list(word_counts.values())
            
            if len(frequencies) < 2:
                return 0.5
            
            # Calculate burstiness using coefficient of variation
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            
            if mean_freq == 0:
                return 0.5
            
            burstiness = std_freq / mean_freq
            
            # Normalize to 0-1 range
            normalized_burstiness = min(1.0, burstiness / 2.0)
            
            return float(normalized_burstiness)
            
        except Exception as e:
            logger.error(f"Error calculating burstiness: {e}")
            return 0.5

class RepetitionPatternAnalyzer:
    """Analyzer for repetitive patterns in text."""
    
    def analyze_repetition_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze various repetition patterns that may indicate AI generation.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of repetition pattern scores
        """
        try:
            sentences = text.split('.')
            words = text.lower().split()
            
            patterns = {}
            
            # Sentence structure repetition
            patterns['sentence_structure_repetition'] = self._calculate_sentence_structure_repetition(sentences)
            
            # N-gram repetition
            patterns['bigram_repetition'] = self._calculate_ngram_repetition(words, 2)
            patterns['trigram_repetition'] = self._calculate_ngram_repetition(words, 3)
            
            # Phrase repetition
            patterns['phrase_repetition'] = self._calculate_phrase_repetition(text)
            
            # Transition word overuse
            patterns['transition_overuse'] = self._calculate_transition_overuse(words)
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing repetition patterns: {e}")
            return {
                'sentence_structure_repetition': 0.0,
                'bigram_repetition': 0.0,
                'trigram_repetition': 0.0,
                'phrase_repetition': 0.0,
                'transition_overuse': 0.0
            }
    
    def _calculate_sentence_structure_repetition(self, sentences: List[str]) -> float:
        """Calculate repetition in sentence structures."""
        if len(sentences) < 3:
            return 0.0
        
        # Simple heuristic: count sentences with similar length patterns
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
        
        length_counts = Counter(lengths)
        max_repetition = max(length_counts.values()) if length_counts else 1
        
        return min(1.0, (max_repetition - 1) / len(lengths))
    
    def _calculate_ngram_repetition(self, words: List[str], n: int) -> float:
        """Calculate n-gram repetition score."""
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        ngram_counts = Counter(ngrams)
        
        total_ngrams = len(ngrams)
        repeated_ngrams = sum(count - 1 for count in ngram_counts.values() if count > 1)
        
        return repeated_ngrams / total_ngrams if total_ngrams > 0 else 0.0
    
    def _calculate_phrase_repetition(self, text: str) -> float:
        """Calculate repetition of common phrases."""
        # Common AI-generated phrases
        ai_phrases = [
            "it is important to note",
            "in conclusion",
            "furthermore",
            "moreover",
            "in addition",
            "on the other hand",
            "it should be noted",
            "as mentioned earlier"
        ]
        
        text_lower = text.lower()
        phrase_count = sum(text_lower.count(phrase) for phrase in ai_phrases)
        
        # Normalize by text length
        words_count = len(text.split())
        return min(1.0, phrase_count / (words_count / 100)) if words_count > 0 else 0.0
    
    def _calculate_transition_overuse(self, words: List[str]) -> float:
        """Calculate overuse of transition words."""
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'nonetheless', 'meanwhile'
        }
        
        transition_count = sum(1 for word in words if word.lower() in transition_words)
        
        return min(1.0, transition_count / (len(words) / 50)) if words else 0.0

class LinguisticMarkerAnalyzer:
    """Analyzer for linguistic markers that indicate AI generation."""
    
    def analyze_linguistic_markers(self, text: str) -> Dict[str, float]:
        """
        Analyze linguistic markers associated with AI-generated content.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of linguistic marker scores
        """
        try:
            markers = {}
            
            # Formality markers
            markers['excessive_formality'] = self._calculate_formality_score(text)
            
            # Hedging language
            markers['hedging_language'] = self._calculate_hedging_score(text)
            
            # Generic language
            markers['generic_language'] = self._calculate_generic_language_score(text)
            
            # Lack of personal experience
            markers['lack_personal_experience'] = self._calculate_personal_experience_score(text)
            
            # Overly structured language
            markers['overly_structured'] = self._calculate_structure_score(text)
            
            return markers
            
        except Exception as e:
            logger.error(f"Error analyzing linguistic markers: {e}")
            return {
                'excessive_formality': 0.0,
                'hedging_language': 0.0,
                'generic_language': 0.0,
                'lack_personal_experience': 0.0,
                'overly_structured': 0.0
            }
    
    def _calculate_formality_score(self, text: str) -> float:
        """Calculate excessive formality score."""
        formal_indicators = [
            'utilize', 'facilitate', 'implement', 'demonstrate', 'establish',
            'furthermore', 'moreover', 'consequently', 'therefore', 'thus'
        ]
        
        words = text.lower().split()
        formal_count = sum(1 for word in words if word in formal_indicators)
        
        return min(1.0, formal_count / (len(words) / 100)) if words else 0.0
    
    def _calculate_hedging_score(self, text: str) -> float:
        """Calculate hedging language score."""
        hedging_words = [
            'might', 'could', 'may', 'perhaps', 'possibly', 'potentially',
            'seems', 'appears', 'tends to', 'generally', 'typically'
        ]
        
        text_lower = text.lower()
        hedging_count = sum(text_lower.count(word) for word in hedging_words)
        
        words_count = len(text.split())
        return min(1.0, hedging_count / (words_count / 50)) if words_count > 0 else 0.0
    
    def _calculate_generic_language_score(self, text: str) -> float:
        """Calculate generic language usage score."""
        generic_phrases = [
            'various aspects', 'different ways', 'important factors',
            'significant impact', 'crucial role', 'essential elements',
            'key components', 'fundamental principles'
        ]
        
        text_lower = text.lower()
        generic_count = sum(text_lower.count(phrase) for phrase in generic_phrases)
        
        sentences_count = len(text.split('.'))
        return min(1.0, generic_count / sentences_count) if sentences_count > 0 else 0.0
    
    def _calculate_personal_experience_score(self, text: str) -> float:
        """Calculate lack of personal experience indicators."""
        personal_indicators = [
            'i think', 'i believe', 'in my opinion', 'from my experience',
            'i have seen', 'i noticed', 'personally', 'my view'
        ]
        
        text_lower = text.lower()
        personal_count = sum(text_lower.count(phrase) for phrase in personal_indicators)
        
        # Return inverse score (higher score = less personal experience)
        sentences_count = len(text.split('.'))
        personal_ratio = personal_count / sentences_count if sentences_count > 0 else 0
        
        return max(0.0, 1.0 - personal_ratio * 5)  # Scale appropriately
    
    def _calculate_structure_score(self, text: str) -> float:
        """Calculate overly structured language score."""
        structure_indicators = [
            'firstly', 'secondly', 'thirdly', 'finally', 'in conclusion',
            'to summarize', 'in summary', 'to begin with', 'next'
        ]
        
        text_lower = text.lower()
        structure_count = sum(text_lower.count(word) for word in structure_indicators)
        
        paragraphs_count = len(text.split('\n\n'))
        return min(1.0, structure_count / paragraphs_count) if paragraphs_count > 0 else 0.0

class EnhancedAIDetectionClassifier:
    """
    Enhanced AI detection classifier with multiple detection approaches
    and comprehensive explainability features.
    """
    
    def __init__(self, model_dir: str = "./models/ai_detection_enhanced"):
        """Initialize the enhanced AI detection classifier."""
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyzers
        self.perplexity_analyzer = PerplexityAnalyzer()
        self.burstiness_analyzer = BurstinessAnalyzer()
        self.repetition_analyzer = RepetitionPatternAnalyzer()
        self.linguistic_analyzer = LinguisticMarkerAnalyzer()
        
        # Models
        self.transformer_models = {}
        self.ensemble_classifiers = {}
        
        # Feature extractors
        self.tokenizers = {}
        
    async def initialize_models(self):
        """Initialize all detection models and analyzers."""
        try:
            logger.info("Initializing enhanced AI detection models...")
            
            if not ML_LIBRARIES_AVAILABLE:
                logger.warning("ML libraries not available, using mock AI detection")
                return
            
            # Initialize perplexity analyzer
            await self.perplexity_analyzer.initialize()
            
            # Initialize transformer models for different approaches
            await self._initialize_transformer_models()
            
            # Initialize ensemble classifiers
            self._initialize_ensemble_classifiers()
            
            logger.info("Enhanced AI detection models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced AI detection models: {e}")
            raise
    
    async def _initialize_transformer_models(self):
        """Initialize various transformer models for detection."""
        model_configs = [
            ("roberta-base", "roberta"),
            ("distilbert-base-uncased", "distilbert"),
            ("microsoft/DialoGPT-medium", "dialogpt")
        ]
        
        for model_name, model_key in model_configs:
            try:
                self.tokenizers[model_key] = AutoTokenizer.from_pretrained(model_name)
                self.transformer_models[model_key] = AutoModel.from_pretrained(model_name)
                logger.info(f"Loaded {model_key} model")
            except Exception as e:
                logger.warning(f"Could not load {model_key} model: {e}")
    
    def _initialize_ensemble_classifiers(self):
        """Initialize ensemble classifiers for different feature types."""
        # Stylometric classifier
        self.ensemble_classifiers['stylometric'] = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        # Linguistic classifier
        self.ensemble_classifiers['linguistic'] = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Perplexity classifier
        self.ensemble_classifiers['perplexity'] = LogisticRegression(
            random_state=42,
            class_weight='balanced'
        )
        
        # Meta-classifier for final ensemble
        self.ensemble_classifiers['meta'] = SVC(
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    
    async def extract_comprehensive_features(self, text: str) -> AIDetectionFeatures:
        """
        Extract comprehensive features for AI detection.
        
        Args:
            text: Input text
            
        Returns:
            Comprehensive feature set
        """
        try:
            # Calculate perplexity
            perplexity_score = self.perplexity_analyzer.calculate_perplexity(text)
            
            # Calculate burstiness
            burstiness_score = self.burstiness_analyzer.calculate_burstiness(text)
            
            # Analyze repetition patterns
            repetition_patterns = self.repetition_analyzer.analyze_repetition_patterns(text)
            
            # Analyze linguistic markers
            linguistic_markers = self.linguistic_analyzer.analyze_linguistic_markers(text)
            
            # Extract transformer embeddings
            transformer_embeddings = await self._extract_transformer_embeddings(text)
            
            # Calculate stylometric anomalies
            stylometric_anomalies = self._calculate_stylometric_anomalies(text)
            
            return AIDetectionFeatures(
                perplexity_score=perplexity_score,
                burstiness_score=burstiness_score,
                repetition_patterns=repetition_patterns,
                linguistic_markers=linguistic_markers,
                stylometric_anomalies=stylometric_anomalies,
                transformer_embeddings=transformer_embeddings
            )
            
        except Exception as e:
            logger.error(f"Error extracting comprehensive features: {e}")
            # Return default features
            return AIDetectionFeatures(
                perplexity_score=50.0,
                burstiness_score=0.5,
                repetition_patterns={},
                linguistic_markers={},
                stylometric_anomalies={},
                transformer_embeddings=np.random.rand(768)
            )
    
    async def _extract_transformer_embeddings(self, text: str) -> np.ndarray:
        """Extract embeddings from transformer models."""
        if not ML_LIBRARIES_AVAILABLE or not self.transformer_models:
            return np.random.rand(768)
        
        try:
            embeddings = []
            
            for model_key, model in self.transformer_models.items():
                tokenizer = self.tokenizers[model_key]
                
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use CLS token or mean pooling
                    if hasattr(outputs, 'last_hidden_state'):
                        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                    else:
                        embedding = outputs.pooler_output.squeeze()
                    
                    embeddings.append(embedding.numpy())
            
            # Concatenate embeddings from different models
            if embeddings:
                return np.concatenate(embeddings)
            else:
                return np.random.rand(768)
                
        except Exception as e:
            logger.error(f"Error extracting transformer embeddings: {e}")
            return np.random.rand(768)
    
    def _calculate_stylometric_anomalies(self, text: str) -> Dict[str, float]:
        """Calculate stylometric anomalies that may indicate AI generation."""
        try:
            anomalies = {}
            
            words = text.split()
            sentences = text.split('.')
            
            # Vocabulary diversity anomaly
            unique_words = len(set(word.lower() for word in words))
            vocab_diversity = unique_words / len(words) if words else 0
            anomalies['low_vocab_diversity'] = max(0, 0.5 - vocab_diversity) * 2
            
            # Sentence length uniformity
            if len(sentences) > 1:
                sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
                if sentence_lengths:
                    length_std = np.std(sentence_lengths)
                    length_mean = np.mean(sentence_lengths)
                    uniformity = 1 - (length_std / (length_mean + 1))
                    anomalies['sentence_uniformity'] = max(0, uniformity - 0.3) / 0.7
                else:
                    anomalies['sentence_uniformity'] = 0
            else:
                anomalies['sentence_uniformity'] = 0
            
            # Punctuation patterns
            punctuation_count = sum(1 for char in text if char in '.,!?;:')
            punctuation_ratio = punctuation_count / len(text) if text else 0
            anomalies['punctuation_anomaly'] = abs(punctuation_ratio - 0.05) * 10
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error calculating stylometric anomalies: {e}")
            return {}
    
    async def detect_ai_content_enhanced(self, text: str) -> Tuple[float, float, DetectionExplanation]:
        """
        Enhanced AI content detection with comprehensive analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (ai_probability, confidence, explanation)
        """
        try:
            # Extract comprehensive features
            features = await self.extract_comprehensive_features(text)
            
            # Calculate individual detection scores
            detection_scores = {}
            
            # Perplexity-based detection
            detection_scores['perplexity'] = self._score_perplexity(features.perplexity_score)
            
            # Burstiness-based detection
            detection_scores['burstiness'] = 1.0 - features.burstiness_score
            
            # Repetition pattern detection
            repetition_score = np.mean(list(features.repetition_patterns.values())) if features.repetition_patterns else 0
            detection_scores['repetition'] = repetition_score
            
            # Linguistic marker detection
            linguistic_score = np.mean(list(features.linguistic_markers.values())) if features.linguistic_markers else 0
            detection_scores['linguistic'] = linguistic_score
            
            # Stylometric anomaly detection
            anomaly_score = np.mean(list(features.stylometric_anomalies.values())) if features.stylometric_anomalies else 0
            detection_scores['stylometric'] = anomaly_score
            
            # Ensemble prediction
            ai_probability = np.mean(list(detection_scores.values()))
            
            # Calculate confidence based on agreement between methods
            score_std = np.std(list(detection_scores.values()))
            confidence = max(0.1, 1.0 - score_std)
            
            # Generate explanation
            explanation = self._generate_explanation(features, detection_scores, ai_probability)
            
            return ai_probability, confidence, explanation
            
        except Exception as e:
            logger.error(f"Error in enhanced AI detection: {e}")
            return 0.5, 0.0, DetectionExplanation(
                primary_indicators=["Error in detection"],
                confidence_factors={"error": 1.0},
                linguistic_evidence={"error": str(e)},
                model_contributions={"error": 1.0},
                risk_assessment="Unknown",
                human_readable_summary=f"Detection failed due to error: {str(e)}"
            )
    
    def _score_perplexity(self, perplexity: float) -> float:
        """Convert perplexity to AI probability score."""
        # Lower perplexity often indicates AI-generated text
        # Normalize perplexity to 0-1 scale
        if perplexity < 20:
            return 0.8  # High AI probability
        elif perplexity < 50:
            return 0.5  # Medium AI probability
        else:
            return 0.2  # Low AI probability
    
    def _generate_explanation(self, features: AIDetectionFeatures, 
                            detection_scores: Dict[str, float], 
                            ai_probability: float) -> DetectionExplanation:
        """Generate comprehensive explanation for the detection result."""
        try:
            # Identify primary indicators
            primary_indicators = []
            confidence_factors = {}
            
            for method, score in detection_scores.items():
                if score > 0.7:
                    primary_indicators.append(f"High {method} score ({score:.2f})")
                confidence_factors[method] = score
            
            # Linguistic evidence
            linguistic_evidence = {
                "perplexity_score": features.perplexity_score,
                "burstiness_score": features.burstiness_score,
                "repetition_patterns": features.repetition_patterns,
                "linguistic_markers": features.linguistic_markers,
                "stylometric_anomalies": features.stylometric_anomalies
            }
            
            # Risk assessment
            if ai_probability > 0.8:
                risk_assessment = "High risk of AI generation"
            elif ai_probability > 0.6:
                risk_assessment = "Moderate risk of AI generation"
            elif ai_probability > 0.4:
                risk_assessment = "Low risk of AI generation"
            else:
                risk_assessment = "Likely human-written"
            
            # Human-readable summary
            summary_parts = []
            
            if features.perplexity_score < 30:
                summary_parts.append("Text shows low perplexity (highly predictable)")
            
            if features.burstiness_score < 0.3:
                summary_parts.append("Low burstiness indicates uniform word distribution")
            
            if any(score > 0.5 for score in features.repetition_patterns.values()):
                summary_parts.append("Repetitive patterns detected")
            
            if any(score > 0.5 for score in features.linguistic_markers.values()):
                summary_parts.append("AI-typical linguistic markers found")
            
            if not summary_parts:
                summary_parts.append("No strong AI indicators detected")
            
            human_readable_summary = ". ".join(summary_parts) + "."
            
            return DetectionExplanation(
                primary_indicators=primary_indicators,
                confidence_factors=confidence_factors,
                linguistic_evidence=linguistic_evidence,
                model_contributions=detection_scores,
                risk_assessment=risk_assessment,
                human_readable_summary=human_readable_summary
            )
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return DetectionExplanation(
                primary_indicators=["Error generating explanation"],
                confidence_factors={"error": 1.0},
                linguistic_evidence={"error": str(e)},
                model_contributions={"error": 1.0},
                risk_assessment="Unknown",
                human_readable_summary="Could not generate explanation due to error."
            )

# Global instance
enhanced_ai_detector = EnhancedAIDetectionClassifier()