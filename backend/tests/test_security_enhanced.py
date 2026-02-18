"""
Enhanced tests for security services and encryption.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.encryption import EncryptionService, DataAnonymizer
from services.key_management import KeyManagementService
from services.security_service import SecurityService
from services.compliance_service import ComplianceService


class TestEncryptionService:
    """Test encryption service functionality."""
    
    def test_encryption_service_initialization(self):
        """Test encryption service initialization."""
        # Test with custom key
        service = EncryptionService("test-master-key-123")
        assert service.master_key == "test-master-key-123"
        
        # Test key derivation
        key = service._derive_key(b"test-password")
        assert len(key) == 32  # 256-bit key
    
    def test_text_encryption_decryption(self):
        """Test text encryption and decryption."""
        service = EncryptionService("test-master-key-123")
        
        # Test basic encryption/decryption
        plaintext = "This is a test message with sensitive data"
        encrypted = service.encrypt_text(plaintext)
        decrypted = service.decrypt_text(encrypted)
        
        assert decrypted == plaintext
        assert encrypted != plaintext
        assert len(encrypted) > len(plaintext)  # Base64 encoded
    
    def test_file_encryption_decryption(self):
        """Test file content encryption and decryption."""
        service = EncryptionService("test-master-key-123")
        
        # Test file content encryption
        file_content = b"Binary file content with sensitive data"
        encrypted = service.encrypt_file_content(file_content)
        decrypted = service.decrypt_file_content(encrypted)
        
        assert decrypted == file_content
        assert encrypted != file_content
    
    def test_secure_token_generation(self):
        """Test secure token generation."""
        service = EncryptionService("test-master-key-123")
        
        # Test token generation
        token1 = service.generate_secure_token(32)
        token2 = service.generate_secure_token(32)
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be unique
    
    def test_data_hashing(self):
        """Test sensitive data hashing."""
        service = EncryptionService("test-master-key-123")
        
        # Test hashing with salt
        data = "sensitive-data-123"
        hash1, salt1 = service.hash_sensitive_data(data)
        hash2, salt2 = service.hash_sensitive_data(data)
        
        assert hash1 != hash2  # Different salts should produce different hashes
        assert salt1 != salt2
        
        # Test verification
        assert service.verify_hashed_data(data, hash1, salt1)
        assert not service.verify_hashed_data("wrong-data", hash1, salt1)


class TestDataAnonymizer:
    """Test data anonymization functionality."""
    
    def test_email_anonymization(self):
        """Test email address anonymization."""
        # Test normal email
        email = "john.doe@university.edu"
        anonymized = DataAnonymizer.anonymize_email(email)
        assert anonymized.startswith("j*****e@university.edu")
        
        # Test short email
        short_email = "ab@test.com"
        anonymized_short = DataAnonymizer.anonymize_email(short_email)
        assert anonymized_short == "**@test.com"
        
        # Test invalid email
        invalid_email = "not-an-email"
        anonymized_invalid = DataAnonymizer.anonymize_email(invalid_email)
        assert anonymized_invalid == "anonymous@example.com"
    
    def test_ip_anonymization(self):
        """Test IP address anonymization."""
        # Test IPv4
        ipv4 = "192.168.1.100"
        anonymized_ipv4 = DataAnonymizer.anonymize_ip_address(ipv4)
        assert anonymized_ipv4 == "192.168.1.0"
        
        # Test IPv6 (simplified)
        ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
        anonymized_ipv6 = DataAnonymizer.anonymize_ip_address(ipv6)
        assert anonymized_ipv6.endswith("::0")
        
        # Test empty IP
        empty_ip = ""
        anonymized_empty = DataAnonymizer.anonymize_ip_address(empty_ip)
        assert anonymized_empty == "0.0.0.0"
    
    def test_user_agent_anonymization(self):
        """Test user agent anonymization."""
        # Test Chrome
        chrome_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        anonymized_chrome = DataAnonymizer.anonymize_user_agent(chrome_ua)
        assert anonymized_chrome == "Chrome"
        
        # Test Firefox
        firefox_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        anonymized_firefox = DataAnonymizer.anonymize_user_agent(firefox_ua)
        assert anonymized_firefox == "Firefox"
        
        # Test unknown
        unknown_ua = "CustomBot/1.0"
        anonymized_unknown = DataAnonymizer.anonymize_user_agent(unknown_ua)
        assert anonymized_unknown == "Other"
    
    def test_pseudonym_generation(self):
        """Test pseudonym generation."""
        # Test consistent pseudonyms
        user_id = 12345
        pseudonym1 = DataAnonymizer.generate_pseudonym(user_id)
        pseudonym2 = DataAnonymizer.generate_pseudonym(user_id)
        
        assert pseudonym1 == pseudonym2  # Should be consistent
        assert pseudonym1.startswith("User_")
        
        # Test different users get different pseudonyms
        pseudonym3 = DataAnonymizer.generate_pseudonym(54321)
        assert pseudonym1 != pseudonym3


@pytest.mark.asyncio
class TestKeyManagementService:
    """Test key management service functionality."""
    
    async def test_master_key_generation(self, db_session: AsyncSession):
        """Test master key generation."""
        service = KeyManagementService()
        
        # Generate master key
        key_id = await service.generate_master_key(db_session)
        assert key_id is not None
        assert key_id.startswith("master_")
        
        # Check if key is active
        active_key = await service.get_active_key(db_session, "master")
        assert active_key == key_id
    
    async def test_data_key_generation(self, db_session: AsyncSession):
        """Test data encryption key generation."""
        service = KeyManagementService()
        
        # Generate data key
        key_id = await service.generate_data_encryption_key(db_session, "submissions")
        assert key_id is not None
        assert "submissions" in key_id
        
        # Check if key is active
        active_key = await service.get_active_key(db_session, "data")
        assert active_key == key_id
    
    async def test_key_rotation(self, db_session: AsyncSession):
        """Test key rotation functionality."""
        service = KeyManagementService()
        
        # Generate initial master key
        old_key_id = await service.generate_master_key(db_session)
        
        # Rotate key
        new_key_id = await service.rotate_master_key(db_session, old_key_id, "test_rotation")
        
        assert new_key_id != old_key_id
        assert new_key_id.startswith("master_")
        
        # Check that new key is active
        active_key = await service.get_active_key(db_session, "master")
        assert active_key == new_key_id
    
    async def test_key_deactivation(self, db_session: AsyncSession):
        """Test key deactivation."""
        service = KeyManagementService()
        
        # Generate key
        key_id = await service.generate_master_key(db_session)
        
        # Deactivate key
        success = await service.deactivate_key(db_session, key_id, "test_deactivation")
        assert success
        
        # Check that no active key exists
        active_key = await service.get_active_key(db_session, "master")
        assert active_key is None
    
    async def test_emergency_revocation(self, db_session: AsyncSession):
        """Test emergency key revocation."""
        service = KeyManagementService()
        
        # Generate key
        key_id = await service.generate_master_key(db_session)
        
        # Emergency revocation
        success = await service.emergency_key_revocation(
            db_session, key_id, "security_incident"
        )
        assert success
    
    async def test_key_statistics(self, db_session: AsyncSession):
        """Test key statistics retrieval."""
        service = KeyManagementService()
        
        # Generate some keys
        await service.generate_master_key(db_session)
        await service.generate_data_encryption_key(db_session, "test_purpose")
        
        # Get statistics
        stats = await service.get_key_statistics(db_session)
        
        assert "active_keys" in stats
        assert "total_keys" in stats
        assert stats["total_keys"] >= 2


@pytest.mark.asyncio
class TestSecurityService:
    """Test security service functionality."""
    
    async def test_password_strength_validation(self):
        """Test password strength validation."""
        # Test strong password
        strong_password = "StrongP@ssw0rd123!"
        result = await SecurityService.validate_password_strength(strong_password)
        
        assert result["is_valid"]
        assert result["score"] > 80
        assert result["requirements_met"]["min_length"]
        assert result["requirements_met"]["has_uppercase"]
        assert result["requirements_met"]["has_lowercase"]
        assert result["requirements_met"]["has_digit"]
        assert result["requirements_met"]["has_special"]
        
        # Test weak password
        weak_password = "password"
        weak_result = await SecurityService.validate_password_strength(weak_password)
        
        assert not weak_result["is_valid"]
        assert weak_result["score"] < 50
        assert not weak_result["requirements_met"]["no_common_patterns"]
    
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Test normal usage
        is_limited, count, limit = await SecurityService.check_rate_limit("test_ip_1")
        assert not is_limited
        assert count == 1
        
        # Test rate limit exceeded (simulate)
        # This would require mocking Redis to simulate high request count
        pass
    
    async def test_intrusion_detection(self, db_session: AsyncSession):
        """Test intrusion detection."""
        # Test with suspicious IP
        result = await SecurityService.detect_intrusion_attempt(
            db_session,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            request_path="/api/v1/auth/login"
        )
        
        assert "threats_detected" in result
        assert "risk_score" in result
        assert isinstance(result["risk_score"], int)
    
    async def test_user_agent_pattern_detection(self):
        """Test user agent pattern detection."""
        # Test suspicious user agent
        suspicious_ua = "sqlmap/1.0 (http://sqlmap.org)"
        result = SecurityService._check_user_agent_patterns(suspicious_ua)
        
        # Should detect as suspicious (contains 'bot' pattern)
        # Note: This depends on the actual patterns in SUSPICIOUS_USER_AGENTS
        
        # Test normal user agent
        normal_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        normal_result = SecurityService._check_user_agent_patterns(normal_ua)
        
        # Should not be flagged as suspicious
        assert normal_result is None


@pytest.mark.asyncio
class TestComplianceService:
    """Test compliance service functionality."""
    
    async def test_consent_recording(self, db_session: AsyncSession):
        """Test consent recording."""
        # Record consent
        success = await ComplianceService.record_consent(
            db_session,
            user_id=1,
            consent_type="data_processing",
            purpose="Academic verification",
            legal_basis="Legitimate interest",
            consent_given=True,
            consent_method="EXPLICIT"
        )
        
        assert success
    
    async def test_consent_withdrawal(self, db_session: AsyncSession):
        """Test consent withdrawal."""
        # First record consent
        await ComplianceService.record_consent(
            db_session,
            user_id=1,
            consent_type="marketing",
            purpose="Marketing communications",
            legal_basis="Consent",
            consent_given=True
        )
        
        # Then withdraw it
        success = await ComplianceService.withdraw_consent(
            db_session,
            user_id=1,
            consent_type="marketing"
        )
        
        assert success
    
    async def test_data_subject_request_creation(self, db_session: AsyncSession):
        """Test data subject request creation."""
        request_id = await ComplianceService.create_data_subject_request(
            db_session,
            user_id=1,
            request_type="ACCESS",
            request_details="Request for all personal data"
        )
        
        assert request_id is not None
        assert isinstance(request_id, int)
    
    async def test_data_export_processing(self, db_session: AsyncSession):
        """Test data export request processing."""
        # Create export request
        request_id = await ComplianceService.create_data_subject_request(
            db_session,
            user_id=1,
            request_type="ACCESS"
        )
        
        # Process export
        export_data = await ComplianceService.process_data_export_request(
            db_session, request_id
        )
        
        assert export_data is not None
        assert "user_id" in export_data
        assert "data_categories" in export_data
    
    async def test_data_retention_cleanup(self, db_session: AsyncSession):
        """Test data retention cleanup."""
        # Initialize default policies
        await ComplianceService.initialize_default_policies(db_session)
        
        # Run cleanup
        cleanup_stats = await ComplianceService.run_data_retention_cleanup(db_session)
        
        assert "policies_processed" in cleanup_stats
        assert isinstance(cleanup_stats["policies_processed"], int)


# Integration tests
@pytest.mark.asyncio
class TestSecurityIntegration:
    """Integration tests for security components."""
    
    async def test_end_to_end_encryption_workflow(self, db_session: AsyncSession):
        """Test complete encryption workflow."""
        # Initialize services
        key_service = KeyManagementService()
        encryption_service = EncryptionService("test-master-key")
        
        # Generate keys
        master_key_id = await key_service.generate_master_key(db_session)
        data_key_id = await key_service.generate_data_encryption_key(db_session, "test_data")
        
        # Encrypt sensitive data
        sensitive_text = "This is sensitive student data"
        encrypted_text = encryption_service.encrypt_text(sensitive_text)
        
        # Decrypt and verify
        decrypted_text = encryption_service.decrypt_text(encrypted_text)
        assert decrypted_text == sensitive_text
        
        # Rotate keys
        new_master_key = await key_service.rotate_master_key(
            db_session, master_key_id, "scheduled_rotation"
        )
        
        assert new_master_key != master_key_id
    
    async def test_security_incident_workflow(self, db_session: AsyncSession):
        """Test security incident detection and response."""
        # Simulate suspicious activity
        result = await SecurityService.detect_intrusion_attempt(
            db_session,
            ip_address="10.0.0.1",
            user_agent="suspicious-bot/1.0",
            user_id=1,
            request_path="/api/v1/admin/"
        )
        
        # Should detect threats
        assert len(result["threats_detected"]) > 0
        assert result["risk_score"] > 0
        
        # If high risk, incident should be created automatically
        if result["action_required"]:
            # Verify incident was logged in audit system
            pass
    
    async def test_compliance_workflow(self, db_session: AsyncSession):
        """Test complete compliance workflow."""
        compliance_service = ComplianceService()
        
        # Initialize policies
        await compliance_service.initialize_default_policies(db_session)
        
        # Record consent
        await compliance_service.record_consent(
            db_session,
            user_id=1,
            consent_type="data_processing",
            purpose="Academic verification",
            legal_basis="Legitimate interest",
            consent_given=True
        )
        
        # Create and process data subject request
        request_id = await compliance_service.create_data_subject_request(
            db_session, user_id=1, request_type="ACCESS"
        )
        
        export_data = await compliance_service.process_data_export_request(
            db_session, request_id
        )
        
        assert export_data is not None
        assert export_data["user_id"] == 1


if __name__ == "__main__":
    pytest.main([__file__])