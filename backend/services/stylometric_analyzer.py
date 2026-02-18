"""
Stylometric feature extraction system for Project Stylos.

This module provides comprehensive stylometric analysis including:
- Lexical richness calculations (TTR, MTLD)
- Sentence length and structure analysis
- Punctuation pattern and frequency analysis
- POS tagging and syntactic feature extraction
"""

import re
import logging
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)


class StylometricAnalyzer:
    """
    Comprehensive stylometric feature extraction for authorship analysis.
    
    This class implements various stylometric measures including lexical richness,
    syntactic patterns, and writing style characteristics.
    """
    
    def __init__(self):
        """Initialize the stylometric analyzer with required resources."""
        self.function_words = {
            'articles': ['a', 'an', 'the'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'to', 'from', 'of', 'about'],
            'pronouns': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'],
            'conjunctions': ['and', 'or', 'but', 'so', 'yet', 'for', 'nor'],
            'auxiliary_verbs': ['is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did']
        }
    
    def extract_features(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive stylometric features from text.
        
        Args:
            text: Input text for analysis
            
        Returns:
            Dictionary containing all extracted stylometric features
        """
        try:
            # Basic preprocessing
            cleaned_text = self._preprocess_text(text)
            sentences = sent_tokenize(cleaned_text)
            words = word_tokenize(cleaned_text.lower())
            
            # Filter out punctuation for word-based analysis
            word_tokens = [word for word in words if word.isalpha()]
            
            # POS tagging
            pos_tags = pos_tag(word_tokens)
            
            # Extract all feature categories
            features = {}
            features.update(self._calculate_lexical_richness(word_tokens))
            features.update(self._analyze_sentence_structure(sentences))
            features.update(self._analyze_punctuation_patterns(text))
            features.update(self._analyze_pos_distribution(pos_tags))
            features.update(self._analyze_function_words(word_tokens))
            features.update(self._calculate_readability_metrics(text, sentences, word_tokens))
            
            logger.info(f"Extracted {len(features)} stylometric features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting stylometric features: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\']', '', text)
        return text.strip()

    def _calculate_lexical_richness(self, words: List[str]) -> Dict[str, float]:
        """
        Calculate various lexical richness measures.
        
        Args:
            words: List of word tokens
            
        Returns:
            Dictionary with lexical richness metrics
        """
        if not words:
            return {
                'type_token_ratio': 0.0,
                'mtld': 0.0,
                'vocabulary_size': 0,
                'hapax_legomena_ratio': 0.0,
                'dis_legomena_ratio': 0.0
            }
        
        word_freq = Counter(words)
        unique_words = len(word_freq)
        total_words = len(words)
        
        # Type-Token Ratio (TTR)
        ttr = unique_words / total_words if total_words > 0 else 0.0
        
        # Measure of Textual Lexical Diversity (MTLD)
        mtld = self._calculate_mtld(words)
        
        # Hapax legomena (words appearing once)
        hapax_count = sum(1 for count in word_freq.values() if count == 1)
        hapax_ratio = hapax_count / total_words if total_words > 0 else 0.0
        
        # Dis legomena (words appearing twice)
        dis_count = sum(1 for count in word_freq.values() if count == 2)
        dis_ratio = dis_count / total_words if total_words > 0 else 0.0
        
        return {
            'type_token_ratio': ttr,
            'mtld': mtld,
            'vocabulary_size': unique_words,
            'hapax_legomena_ratio': hapax_ratio,
            'dis_legomena_ratio': dis_ratio
        }

    def _calculate_mtld(self, words: List[str], threshold: float = 0.72) -> float:
        """
        Calculate Measure of Textual Lexical Diversity (MTLD).
        
        Args:
            words: List of word tokens
            threshold: TTR threshold for segment calculation
            
        Returns:
            MTLD score
        """
        if len(words) < 50:
            return 0.0
        
        def calculate_factor_length(word_list: List[str], start_idx: int, direction: int) -> int:
            """Calculate length of factor in given direction."""
            current_ttr = 1.0
            unique_words = set()
            factor_length = 0
            idx = start_idx
            
            while 0 <= idx < len(word_list) and current_ttr >= threshold:
                word = word_list[idx]
                unique_words.add(word)
                factor_length += 1
                current_ttr = len(unique_words) / factor_length
                idx += direction
            
            return factor_length
        
        # Forward calculation
        forward_factors = []
        i = 0
        while i < len(words):
            factor_length = calculate_factor_length(words, i, 1)
            if factor_length > 0:
                forward_factors.append(factor_length)
                i += factor_length
            else:
                break
        
        # Backward calculation
        backward_factors = []
        i = len(words) - 1
        while i >= 0:
            factor_length = calculate_factor_length(words, i, -1)
            if factor_length > 0:
                backward_factors.append(factor_length)
                i -= factor_length
            else:
                break
        
        # Calculate MTLD
        all_factors = forward_factors + backward_factors
        if not all_factors:
            return 0.0
        
        return len(words) / (len(all_factors) / 2)

    def _analyze_sentence_structure(self, sentences: List[str]) -> Dict[str, float]:
        """
        Analyze sentence length and structure patterns.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Dictionary with sentence structure metrics
        """
        if not sentences:
            return {
                'avg_sentence_length': 0.0,
                'sentence_length_variance': 0.0,
                'short_sentence_ratio': 0.0,
                'long_sentence_ratio': 0.0,
                'sentence_count': 0
            }
        
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        avg_length = np.mean(sentence_lengths)
        variance = np.var(sentence_lengths)
        
        # Short sentences (< 10 words) and long sentences (> 25 words)
        short_count = sum(1 for length in sentence_lengths if length < 10)
        long_count = sum(1 for length in sentence_lengths if length > 25)
        
        total_sentences = len(sentences)
        short_ratio = short_count / total_sentences
        long_ratio = long_count / total_sentences
        
        return {
            'avg_sentence_length': float(avg_length),
            'sentence_length_variance': float(variance),
            'short_sentence_ratio': short_ratio,
            'long_sentence_ratio': long_ratio,
            'sentence_count': total_sentences
        }

    def _analyze_punctuation_patterns(self, text: str) -> Dict[str, float]:
        """
        Analyze punctuation usage patterns.
        
        Args:
            text: Original text with punctuation
            
        Returns:
            Dictionary with punctuation metrics
        """
        total_chars = len(text)
        if total_chars == 0:
            return {
                'comma_frequency': 0.0,
                'period_frequency': 0.0,
                'exclamation_frequency': 0.0,
                'question_frequency': 0.0,
                'semicolon_frequency': 0.0,
                'colon_frequency': 0.0,
                'dash_frequency': 0.0,
                'quotation_frequency': 0.0,
                'total_punctuation_ratio': 0.0
            }
        
        punctuation_counts = {
            'comma': text.count(','),
            'period': text.count('.'),
            'exclamation': text.count('!'),
            'question': text.count('?'),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'dash': text.count('-') + text.count('—'),
            'quotation': text.count('"') + text.count("'")
        }
        
        total_punctuation = sum(punctuation_counts.values())
        
        return {
            'comma_frequency': punctuation_counts['comma'] / total_chars,
            'period_frequency': punctuation_counts['period'] / total_chars,
            'exclamation_frequency': punctuation_counts['exclamation'] / total_chars,
            'question_frequency': punctuation_counts['question'] / total_chars,
            'semicolon_frequency': punctuation_counts['semicolon'] / total_chars,
            'colon_frequency': punctuation_counts['colon'] / total_chars,
            'dash_frequency': punctuation_counts['dash'] / total_chars,
            'quotation_frequency': punctuation_counts['quotation'] / total_chars,
            'total_punctuation_ratio': total_punctuation / total_chars
        }

    def _analyze_pos_distribution(self, pos_tags: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Analyze part-of-speech tag distribution.
        
        Args:
            pos_tags: List of (word, POS_tag) tuples
            
        Returns:
            Dictionary with POS distribution metrics
        """
        if not pos_tags:
            return {
                'noun_ratio': 0.0,
                'verb_ratio': 0.0,
                'adjective_ratio': 0.0,
                'adverb_ratio': 0.0,
                'pronoun_ratio': 0.0,
                'preposition_ratio': 0.0,
                'conjunction_ratio': 0.0,
                'determiner_ratio': 0.0
            }
        
        pos_counts = Counter(tag for _, tag in pos_tags)
        total_tags = len(pos_tags)
        
        # Group POS tags into major categories
        noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
        verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        adj_tags = ['JJ', 'JJR', 'JJS']
        adv_tags = ['RB', 'RBR', 'RBS']
        pronoun_tags = ['PRP', 'PRP$', 'WP', 'WP$']
        prep_tags = ['IN']
        conj_tags = ['CC']
        det_tags = ['DT', 'WDT']
        
        def calculate_ratio(tag_list: List[str]) -> float:
            return sum(pos_counts.get(tag, 0) for tag in tag_list) / total_tags
        
        return {
            'noun_ratio': calculate_ratio(noun_tags),
            'verb_ratio': calculate_ratio(verb_tags),
            'adjective_ratio': calculate_ratio(adj_tags),
            'adverb_ratio': calculate_ratio(adv_tags),
            'pronoun_ratio': calculate_ratio(pronoun_tags),
            'preposition_ratio': calculate_ratio(prep_tags),
            'conjunction_ratio': calculate_ratio(conj_tags),
            'determiner_ratio': calculate_ratio(det_tags)
        }

    def _analyze_function_words(self, words: List[str]) -> Dict[str, float]:
        """
        Analyze function word usage patterns.
        
        Args:
            words: List of word tokens
            
        Returns:
            Dictionary with function word metrics
        """
        if not words:
            return {
                'article_frequency': 0.0,
                'preposition_frequency': 0.0,
                'pronoun_frequency': 0.0,
                'conjunction_frequency': 0.0,
                'auxiliary_verb_frequency': 0.0,
                'function_word_ratio': 0.0
            }
        
        total_words = len(words)
        function_word_counts = defaultdict(int)
        total_function_words = 0
        
        for word in words:
            for category, word_list in self.function_words.items():
                if word.lower() in word_list:
                    function_word_counts[category] += 1
                    total_function_words += 1
        
        return {
            'article_frequency': function_word_counts['articles'] / total_words,
            'preposition_frequency': function_word_counts['prepositions'] / total_words,
            'pronoun_frequency': function_word_counts['pronouns'] / total_words,
            'conjunction_frequency': function_word_counts['conjunctions'] / total_words,
            'auxiliary_verb_frequency': function_word_counts['auxiliary_verbs'] / total_words,
            'function_word_ratio': total_function_words / total_words
        }

    def _calculate_readability_metrics(self, text: str, sentences: List[str], words: List[str]) -> Dict[str, float]:
        """
        Calculate readability and complexity metrics.
        
        Args:
            text: Original text
            sentences: List of sentences
            words: List of word tokens
            
        Returns:
            Dictionary with readability metrics
        """
        if not sentences or not words:
            return {
                'flesch_reading_ease': 0.0,
                'flesch_kincaid_grade': 0.0,
                'avg_word_length': 0.0,
                'complex_word_ratio': 0.0
            }
        
        # Basic counts
        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(word) for word in words)
        
        # Average word length
        avg_word_length = np.mean([len(word) for word in words])
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)
        complex_word_ratio = complex_words / num_words if num_words > 0 else 0.0
        
        # Flesch Reading Ease
        if num_sentences > 0 and num_words > 0:
            flesch_ease = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (num_syllables / num_words))
            flesch_grade = 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59
        else:
            flesch_ease = 0.0
            flesch_grade = 0.0
        
        return {
            'flesch_reading_ease': flesch_ease,
            'flesch_kincaid_grade': flesch_grade,
            'avg_word_length': float(avg_word_length),
            'complex_word_ratio': complex_word_ratio
        }
    
    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word.
        
        Args:
            word: Input word
            
        Returns:
            Estimated syllable count
        """
        word = word.lower()
        if not word:
            return 0
        
        # Remove common suffixes that don't add syllables
        word = re.sub(r'[^a-z]', '', word)
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Ensure at least one syllable
        return max(1, syllable_count)
    
    def compare_profiles(self, profile1: Dict[str, Any], profile2: Dict[str, Any]) -> Dict[str, float]:
        """
        Compare two stylometric profiles and calculate similarity metrics.
        
        Args:
            profile1: First stylometric profile
            profile2: Second stylometric profile
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Get common features
            common_features = set(profile1.keys()) & set(profile2.keys())
            
            if not common_features:
                return {'similarity_score': 0.0, 'feature_count': 0}
            
            # Calculate feature-wise differences
            differences = []
            for feature in common_features:
                val1 = profile1[feature]
                val2 = profile2[feature]
                
                # Handle numeric values
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Normalize difference by the maximum value to get relative difference
                    max_val = max(abs(val1), abs(val2), 1e-10)  # Avoid division by zero
                    diff = abs(val1 - val2) / max_val
                    differences.append(diff)
            
            if not differences:
                return {'similarity_score': 0.0, 'feature_count': 0}
            
            # Calculate overall similarity (1 - average normalized difference)
            avg_difference = np.mean(differences)
            similarity_score = max(0.0, 1.0 - avg_difference)
            
            return {
                'similarity_score': float(similarity_score),
                'feature_count': len(differences),
                'avg_difference': float(avg_difference)
            }
            
        except Exception as e:
            logger.error(f"Error comparing stylometric profiles: {str(e)}")
            return {'similarity_score': 0.0, 'feature_count': 0, 'error': str(e)}


# Create singleton instance
stylometric_analyzer = StylometricAnalyzer()
