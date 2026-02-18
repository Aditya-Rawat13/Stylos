"""
Authorship Score Validator - Ensures no 0.0% scores ever get saved
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

class AuthorshipScoreValidator:
    """Validator to ensure authorship scores are never 0.0"""
    
    @staticmethod
    def validate_and_fix_score(authorship_score: float, ai_probability: float) -> float:
        """
        Validate authorship score and fix if it's 0.0
        
        Args:
            authorship_score: Original authorship score
            ai_probability: AI detection probability (0.0 to 1.0)
            
        Returns:
            Valid authorship score (guaranteed > 0.0)
        """
        # If score is already valid and non-zero, return it
        if authorship_score > 0.0 and authorship_score <= 1.0:
            return authorship_score
        
        # Calculate replacement score based on AI probability
        if ai_probability <= 0.15:
            replacement_score = 0.90  # Very confident human writing
        elif ai_probability <= 0.30:
            replacement_score = 0.75  # Likely human writing
        elif ai_probability <= 0.50:
            replacement_score = 0.60  # Moderate confidence human
        elif ai_probability <= 0.70:
            replacement_score = 0.45  # Lower confidence human
        elif ai_probability <= 0.85:
            replacement_score = 0.30  # Possible AI but still some human elements
        else:
            replacement_score = 0.15  # Likely AI but never zero
        
        logger.warning(f"VALIDATOR: Fixed invalid authorship score {authorship_score} -> {replacement_score} (AI: {ai_probability:.2f})")
        return replacement_score
    
    @staticmethod
    def get_confidence_level(authorship_score: float) -> str:
        """Get confidence level description"""
        if authorship_score >= 0.8:
            return "Very High"
        elif authorship_score >= 0.6:
            return "High"
        elif authorship_score >= 0.4:
            return "Medium"
        elif authorship_score >= 0.2:
            return "Low"
        else:
            return "Very Low"

# Global validator instance
authorship_validator = AuthorshipScoreValidator()
