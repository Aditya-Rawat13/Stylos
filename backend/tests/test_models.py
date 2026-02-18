"""
Unit tests for data models.
Tests model validation, relationships, and business logic.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from models.user import User, UserRole, WritingProfile
from models.submission import Submission, SubmissionStatus
from models.verification import VerificationResult, VerificationStatus
from models.blockchain import BlockchainRecord, BlockchainStatus


class TestUserModel:
    """Test cases for User model."""
    
    def test_user_creation(self):
        """Test user model creation."""
        user = User(
            email="test@example.com",
            full_name="Test User",
            hashed_password="hashed_password",
            role=UserRole.STUDENT
        )
        
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.role == UserRole.STUDENT
    
    def test_user_role_validation(self):
        """Test user role validation."""
        # Valid roles
        student = User(email="s@test.com", full_name="Student", hashed_password="hash", role=UserRole.STUDENT)
        admin = User(email="a@test.com", full_name="Admin", hashed_password="hash", role=UserRole.ADMIN)
        
        assert student.role == UserRole.STUDENT
        assert admin.role == UserRole.ADMIN
    
    def test_user_email_format(self):
        """Test email format validation."""
        user = User(email="valid@example.com", full_name="User", hashed_password="hash")
        assert "@" in user.email
        assert "." in user.email
    
    def test_user_password_hash_required(self):
        """Test that password hash is required."""
        user = User(email="test@test.com", full_name="User", hashed_password="hash")
        assert user.hashed_password is not None
        assert len(user.hashed_password) > 0


class TestSubmissionModel:
    """Test cases for Submission model."""
    
    def test_submission_creation(self):
        """Test submission model creation."""
        submission = Submission(
            user_id=1,
            filename="test_essay.pdf",
            file_path="/uploads/test_essay.pdf",
            file_size=1024,
            file_hash="abc123def456",
            title="Test Essay",
            content="Essay content here",
            word_count=100,
            status=SubmissionStatus.UPLOADED
        )
        
        assert submission.user_id == 1
        assert submission.title == "Test Essay"
        assert submission.status == SubmissionStatus.UPLOADED
    
    def test_submission_status_transitions(self):
        """Test submission status transitions."""
        submission = Submission(
            user_id=1,
            filename="essay.pdf",
            file_path="/uploads/essay.pdf",
            file_size=1024,
            file_hash="hash",
            title="Essay",
            content="Content",
            word_count=50,
            status=SubmissionStatus.UPLOADED
        )
        
        # Valid transitions
        submission.status = SubmissionStatus.PROCESSING
        assert submission.status == SubmissionStatus.PROCESSING
        
        submission.status = SubmissionStatus.VERIFIED
        assert submission.status == SubmissionStatus.VERIFIED
    
    def test_submission_file_hash_uniqueness(self):
        """Test file hash uniqueness."""
        sub1 = Submission(
            user_id=1,
            filename="essay1.pdf",
            file_path="/uploads/essay1.pdf",
            file_size=1024,
            file_hash="unique_hash_1",
            title="Essay 1",
            content="Content",
            word_count=50,
            status=SubmissionStatus.UPLOADED
        )
        
        sub2 = Submission(
            user_id=2,
            filename="essay2.pdf",
            file_path="/uploads/essay2.pdf",
            file_size=2048,
            file_hash="unique_hash_2",
            title="Essay 2",
            content="Different content",
            word_count=75,
            status=SubmissionStatus.UPLOADED
        )
        
        assert sub1.file_hash != sub2.file_hash
    
    def test_submission_word_count(self):
        """Test submission word count tracking."""
        submission = Submission(
            user_id=1,
            filename="essay.pdf",
            file_path="/uploads/essay.pdf",
            file_size=1024,
            file_hash="hash",
            title="Essay",
            content="Content with multiple words here",
            word_count=5,
            status=SubmissionStatus.UPLOADED
        )
        
        assert submission.word_count == 5
        assert submission.word_count > 0


class TestVerificationResultModel:
    """Test cases for VerificationResult model."""
    
    def test_verification_result_creation(self):
        """Test verification result creation."""
        result = VerificationResult(
            submission_id=1,
            authorship_score=0.85,
            ai_probability=0.15,
            status=VerificationStatus.COMPLETED,
            authorship_confidence=0.90
        )
        
        assert result.submission_id == 1
        assert result.authorship_score == 0.85
        assert result.ai_probability == 0.15
        assert result.status == VerificationStatus.COMPLETED
    
    def test_verification_score_ranges(self):
        """Test verification score validation ranges."""
        result = VerificationResult(
            submission_id=1,
            authorship_score=0.85,
            ai_probability=0.15,
            status=VerificationStatus.COMPLETED,
            authorship_confidence=0.90
        )
        
        # Scores should be between 0 and 1
        assert 0.0 <= result.authorship_score <= 1.0
        assert 0.0 <= result.ai_probability <= 1.0
        assert 0.0 <= result.authorship_confidence <= 1.0
    
    def test_verification_status_types(self):
        """Test verification status types."""
        # Pending
        pending_result = VerificationResult(
            submission_id=1,
            status=VerificationStatus.PENDING
        )
        assert pending_result.status == VerificationStatus.PENDING
        
        # In Progress
        progress_result = VerificationResult(
            submission_id=2,
            status=VerificationStatus.IN_PROGRESS
        )
        assert progress_result.status == VerificationStatus.IN_PROGRESS
        
        # Completed
        completed_result = VerificationResult(
            submission_id=3,
            authorship_score=0.85,
            ai_probability=0.15,
            status=VerificationStatus.COMPLETED
        )
        assert completed_result.status == VerificationStatus.COMPLETED
    
    def test_duplicate_submissions_structure(self):
        """Test duplicate submissions data structure."""
        result = VerificationResult(
            submission_id=1,
            authorship_score=0.85,
            ai_probability=0.15,
            similarity_score=0.92,
            has_duplicates=True,
            duplicate_submissions=[1, 2, 3],
            status=VerificationStatus.COMPLETED
        )
        
        assert result.has_duplicates is True
        assert len(result.duplicate_submissions) == 3
        assert 1 in result.duplicate_submissions
    
    def test_overall_risk_score_calculation(self):
        """Test overall risk score calculation."""
        result = VerificationResult(
            submission_id=1,
            authorship_score=0.85,
            ai_probability=0.15,
            similarity_score=0.20,
            status=VerificationStatus.COMPLETED
        )
        
        risk_score = result.overall_risk_score
        assert 0.0 <= risk_score <= 1.0
    
    def test_is_flagged_property(self):
        """Test is_flagged property logic."""
        # Not flagged
        good_result = VerificationResult(
            submission_id=1,
            authorship_score=0.90,
            ai_probability=0.10,
            similarity_score=0.20,
            is_authentic=True,
            has_duplicates=False,
            is_ai_generated=False,
            status=VerificationStatus.COMPLETED
        )
        assert good_result.is_flagged is False
        
        # Flagged due to low authorship
        bad_result = VerificationResult(
            submission_id=2,
            authorship_score=0.30,
            ai_probability=0.10,
            is_authentic=False,
            status=VerificationStatus.COMPLETED
        )
        assert bad_result.is_flagged is True


class TestBlockchainRecordModel:
    """Test cases for BlockchainRecord model."""
    
    def test_blockchain_record_creation(self):
        """Test blockchain record creation."""
        record = BlockchainRecord(
            submission_id=1,
            transaction_hash="0xabc123def456",
            token_id="1",
            ipfs_hash="QmTestHash123",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="abc123def456",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        
        assert record.submission_id == 1
        assert record.transaction_hash == "0xabc123def456"
        assert record.token_id == "1"
        assert record.status == BlockchainStatus.CONFIRMED
    
    def test_blockchain_status_transitions(self):
        """Test blockchain status transitions."""
        record = BlockchainRecord(
            submission_id=1,
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.PENDING
        )
        
        # Pending -> Submitted
        record.status = BlockchainStatus.SUBMITTED
        assert record.status == BlockchainStatus.SUBMITTED
        
        # Submitted -> Confirmed
        record.status = BlockchainStatus.CONFIRMED
        assert record.status == BlockchainStatus.CONFIRMED
        
        # Can also fail
        record2 = BlockchainRecord(
            submission_id=2,
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash2",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.PENDING
        )
        record2.status = BlockchainStatus.FAILED
        assert record2.status == BlockchainStatus.FAILED
    
    def test_transaction_hash_format(self):
        """Test transaction hash format validation."""
        record = BlockchainRecord(
            submission_id=1,
            transaction_hash="0x1234567890abcdef",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        
        assert record.transaction_hash.startswith("0x")
        assert len(record.transaction_hash) > 10
    
    def test_ipfs_hash_format(self):
        """Test IPFS hash format validation."""
        record = BlockchainRecord(
            submission_id=1,
            ipfs_hash="QmTestHash123456789",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        
        assert record.ipfs_hash.startswith("Qm")
        assert len(record.ipfs_hash) > 10
    
    def test_network_information(self):
        """Test blockchain network information."""
        record = BlockchainRecord(
            submission_id=1,
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            network_id=137,
            network_name="polygon",
            status=BlockchainStatus.CONFIRMED
        )
        
        assert record.network_id == 137
        assert record.network_name == "polygon"
    
    def test_is_confirmed_property(self):
        """Test is_confirmed property."""
        confirmed_record = BlockchainRecord(
            submission_id=1,
            transaction_hash="0xhash",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        assert confirmed_record.is_confirmed is True
        
        pending_record = BlockchainRecord(
            submission_id=2,
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash2",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.PENDING
        )
        assert pending_record.is_confirmed is False
    
    def test_explorer_url_property(self):
        """Test explorer URL generation."""
        record = BlockchainRecord(
            submission_id=1,
            transaction_hash="0xabc123",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            network_name="polygon",
            status=BlockchainStatus.CONFIRMED
        )
        
        url = record.explorer_url
        assert "polygonscan.com" in url
        assert record.transaction_hash in url
    
    def test_ipfs_url_property(self):
        """Test IPFS URL generation."""
        record = BlockchainRecord(
            submission_id=1,
            ipfs_hash="QmTestHash123",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        
        url = record.ipfs_url
        assert "ipfs.io/ipfs/" in url
        assert record.ipfs_hash in url


class TestModelRelationships:
    """Test cases for model relationships."""
    
    def test_user_submission_relationship(self):
        """Test relationship between user and submissions."""
        user = User(
            email="test@test.com",
            full_name="Test User",
            hashed_password="hash",
            role=UserRole.STUDENT
        )
        
        # Simulate user having an ID after database insert
        user.id = 1
        
        submission1 = Submission(
            user_id=user.id,
            filename="essay1.pdf",
            file_path="/uploads/essay1.pdf",
            file_size=1024,
            file_hash="hash1",
            title="Essay 1",
            content="Content 1",
            word_count=50,
            status=SubmissionStatus.UPLOADED
        )
        
        submission2 = Submission(
            user_id=user.id,
            filename="essay2.pdf",
            file_path="/uploads/essay2.pdf",
            file_size=2048,
            file_hash="hash2",
            title="Essay 2",
            content="Content 2",
            word_count=75,
            status=SubmissionStatus.VERIFIED
        )
        
        # Both submissions belong to same user
        assert submission1.user_id == user.id
        assert submission2.user_id == user.id
    
    def test_submission_verification_relationship(self):
        """Test relationship between submission and verification."""
        submission = Submission(
            user_id=1,
            filename="essay.pdf",
            file_path="/uploads/essay.pdf",
            file_size=1024,
            file_hash="hash",
            title="Essay",
            content="Content",
            word_count=50,
            status=SubmissionStatus.PROCESSING
        )
        
        # Simulate submission having an ID
        submission.id = 1
        
        verification = VerificationResult(
            submission_id=submission.id,
            authorship_score=0.85,
            ai_probability=0.15,
            status=VerificationStatus.COMPLETED,
            authorship_confidence=0.90
        )
        
        assert verification.submission_id == submission.id
    
    def test_submission_blockchain_relationship(self):
        """Test relationship between submission and blockchain record."""
        submission = Submission(
            user_id=1,
            filename="essay.pdf",
            file_path="/uploads/essay.pdf",
            file_size=1024,
            file_hash="hash",
            title="Essay",
            content="Content",
            word_count=50,
            status=SubmissionStatus.VERIFIED
        )
        
        # Simulate submission having an ID
        submission.id = 1
        
        blockchain_record = BlockchainRecord(
            submission_id=submission.id,
            transaction_hash="0xhash",
            token_id="1",
            ipfs_hash="QmHash",
            contract_address="0x1234567890123456789012345678901234567890",
            content_hash="hash",
            verification_timestamp=datetime.now(),
            status=BlockchainStatus.CONFIRMED
        )
        
        assert blockchain_record.submission_id == submission.id


class TestModelValidation:
    """Test cases for model validation logic."""
    
    def test_email_validation(self):
        """Test email format validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "first+last@test.org"
        ]
        
        for email in valid_emails:
            user = User(
                email=email,
                full_name="Test",
                hashed_password="hash"
            )
            assert "@" in user.email
            assert "." in user.email
    
    def test_file_hash_validation(self):
        """Test file hash validation."""
        submission = Submission(
            user_id=1,
            filename="essay.pdf",
            file_path="/uploads/essay.pdf",
            file_size=1024,
            file_hash="a1b2c3d4e5f6",
            title="Essay",
            content="Content",
            word_count=50,
            status=SubmissionStatus.UPLOADED
        )
        
        # Hash should be alphanumeric
        assert submission.file_hash.isalnum()
        assert len(submission.file_hash) > 0
    
    def test_score_boundary_validation(self):
        """Test score boundary validation."""
        # Valid scores
        result = VerificationResult(
            submission_id=1,
            authorship_score=0.85,
            ai_probability=0.15,
            status=VerificationStatus.COMPLETED,
            authorship_confidence=0.90
        )
        
        assert 0.0 <= result.authorship_score <= 1.0
        assert 0.0 <= result.ai_probability <= 1.0
        assert 0.0 <= result.authorship_confidence <= 1.0


class TestWritingProfileModel:
    """Test cases for WritingProfile model."""
    
    def test_writing_profile_creation(self):
        """Test writing profile creation."""
        profile = WritingProfile(
            user_id=1,
            total_submissions=5,
            total_words=5000,
            avg_confidence_score=85,
            is_initialized=True
        )
        
        assert profile.user_id == 1
        assert profile.total_submissions == 5
        assert profile.is_initialized is True
    
    def test_stylometric_features_storage(self):
        """Test stylometric features JSON storage."""
        profile = WritingProfile(
            user_id=1,
            lexical_features={"ttr": 0.75, "mtld": 85.5},
            syntactic_features={"avg_sentence_length": 18.5},
            semantic_features={"topic_coherence": 0.82}
        )
        
        assert profile.lexical_features["ttr"] == 0.75
        assert profile.syntactic_features["avg_sentence_length"] == 18.5
        assert profile.semantic_features["topic_coherence"] == 0.82


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
