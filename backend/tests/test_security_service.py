"""
Unit tests for security service.
Tests encryption, input validation, and security measures.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch
from services.security_service import SecurityService
from utils.validation import sanitize_input, validate_file_type, check_sql_injection


class TestSecurityService:
    """Test cases for SecurityService."""
    
    @pytest.fixture
    def security_service(self):
        """Create security service instance."""
        return SecurityService()
    
    def test_input_sanitization_xss(self, security_service):
        """Test XSS attack prevention."""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<iframe src='malicious.com'></iframe>"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = security_service.sanitize_input(malicious_input)
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "<iframe>" not in sanitized.lower()
    
    def test_input_sanitization_sql_injection(self, security_service):
        """Test SQL injection prevention."""
        malicious_inputs = [
            "'; DROP TABLE students; --",
            "1' OR '1'='1",
            "admin'--",
            "1; DELETE FROM users"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = security_service.sanitize_input(malicious_input)
            # Should escape or remove SQL special characters
            assert "DROP TABLE" not in sanitized.upper() or "'" not in sanitized
    
    def test_file_type_validation_allowed(self, security_service):
        """Test file type validation for allowed types."""
        allowed_files = [
            ("essay.pdf", "application/pdf"),
            ("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("text.txt", "text/plain")
        ]
        
        for filename, mimetype in allowed_files:
            assert security_service.validate_file_type(filename, mimetype) is True
    
    def test_file_type_validation_blocked(self, security_service):
        """Test file type validation for blocked types."""
        blocked_files = [
            ("malware.exe", "application/x-msdownload"),
            ("script.js", "application/javascript"),
            ("virus.bat", "application/x-bat"),
            ("hack.sh", "application/x-sh")
        ]
        
        for filename, mimetype in blocked_files:
            assert security_service.validate_file_type(filename, mimetype) is False
    
    def test_file_size_validation(self, security_service):
        """Test file size validation."""
        # Valid size (5MB)
        assert security_service.validate_file_size(5 * 1024 * 1024) is True
        
        # Too large (100MB)
        assert security_service.validate_file_size(100 * 1024 * 1024) is False
        
        # Empty file
        assert security_service.validate_file_size(0) is False
    
    def test_rate_limiting_check(self, security_service):
        """Test rate limiting functionality."""
        user_id = "user123"
        
        # First few requests should pass
        for i in range(10):
            assert security_service.check_rate_limit(user_id) is True
        
        # After limit, should be blocked
        for i in range(100):
            security_service.check_rate_limit(user_id)
        
        # Should eventually hit rate limit
        assert security_service.is_rate_limited(user_id) is True
    
    def test_csrf_token_generation(self, security_service):
        """Test CSRF token generation."""
        token1 = security_service.generate_csrf_token()
        token2 = security_service.generate_csrf_token()
        
        assert token1 is not None
        assert token2 is not None
        assert token1 != token2
        assert len(token1) > 20
    
    def test_csrf_token_validation(self, security_service):
        """Test CSRF token validation."""
        token = security_service.generate_csrf_token()
        session_id = "session123"
        
        # Store token
        security_service.store_csrf_token(session_id, token)
        
        # Valid token should pass
        assert security_service.validate_csrf_token(session_id, token) is True
        
        # Invalid token should fail
        assert security_service.validate_csrf_token(session_id, "invalid_token") is False
    
    def test_content_encryption(self, security_service):
        """Test content encryption."""
        original_content = "This is sensitive essay content that needs encryption."
        
        encrypted = security_service.encrypt_content(original_content)
        
        assert encrypted != original_content
        assert len(encrypted) > 0
        assert isinstance(encrypted, (str, bytes))
    
    def test_content_decryption(self, security_service):
        """Test content decryption."""
        original_content = "This is sensitive essay content."
        
        encrypted = security_service.encrypt_content(original_content)
        decrypted = security_service.decrypt_content(encrypted)
        
        assert decrypted == original_content
    
    def test_hash_generation_consistency(self, security_service):
        """Test hash generation consistency."""
        content = "Test essay content for hashing"
        
        hash1 = security_service.generate_content_hash(content)
        hash2 = security_service.generate_content_hash(content)
        
        # Same content should produce same hash
        assert hash1 == hash2
    
    def test_hash_generation_uniqueness(self, security_service):
        """Test hash generation uniqueness."""
        content1 = "First essay content"
        content2 = "Second essay content"
        
        hash1 = security_service.generate_content_hash(content1)
        hash2 = security_service.generate_content_hash(content2)
        
        # Different content should produce different hashes
        assert hash1 != hash2
    
    def test_password_complexity_validation(self, security_service):
        """Test password complexity requirements."""
        weak_passwords = [
            "short",
            "12345678",
            "password",
            "abcdefgh",
            "ABCDEFGH",
            "12345678"
        ]
        
        strong_passwords = [
            "SecurePass123!",
            "MyP@ssw0rd2024",
            "C0mpl3x!Pass",
            "Str0ng&Secure"
        ]
        
        for password in weak_passwords:
            assert security_service.validate_password_complexity(password) is False
        
        for password in strong_passwords:
            assert security_service.validate_password_complexity(password) is True
    
    def test_session_token_generation(self, security_service):
        """Test secure session token generation."""
        token1 = security_service.generate_session_token()
        token2 = security_service.generate_session_token()
        
        assert token1 != token2
        assert len(token1) >= 32
        assert len(token2) >= 32
    
    def test_ip_address_validation(self, security_service):
        """Test IP address validation."""
        valid_ips = [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "8.8.8.8"
        ]
        
        invalid_ips = [
            "999.999.999.999",
            "not.an.ip.address",
            "192.168.1",
            "192.168.1.1.1"
        ]
        
        for ip in valid_ips:
            assert security_service.validate_ip_address(ip) is True
        
        for ip in invalid_ips:
            assert security_service.validate_ip_address(ip) is False
    
    def test_suspicious_activity_detection(self, security_service):
        """Test suspicious activity detection."""
        # Normal activity
        assert security_service.detect_suspicious_activity(
            user_id="user123",
            action="view_submission",
            frequency=5
        ) is False
        
        # Suspicious activity (too many requests)
        assert security_service.detect_suspicious_activity(
            user_id="user123",
            action="download_submission",
            frequency=1000
        ) is True
    
    def test_audit_log_creation(self, security_service):
        """Test security audit log creation."""
        log_entry = security_service.create_audit_log(
            user_id="user123",
            action="login",
            ip_address="192.168.1.1",
            success=True
        )
        
        assert log_entry is not None
        assert log_entry["user_id"] == "user123"
        assert log_entry["action"] == "login"
        assert log_entry["success"] is True
    
    def test_data_anonymization(self, security_service):
        """Test data anonymization for analytics."""
        sensitive_data = {
            "email": "student@university.edu",
            "name": "John Doe",
            "student_id": "12345",
            "submission_content": "Essay content here"
        }
        
        anonymized = security_service.anonymize_data(sensitive_data)
        
        assert anonymized["email"] != sensitive_data["email"]
        assert anonymized["name"] != sensitive_data["name"]
        assert "submission_content" not in anonymized or anonymized["submission_content"] != sensitive_data["submission_content"]
    
    def test_secure_random_generation(self, security_service):
        """Test cryptographically secure random generation."""
        random1 = security_service.generate_secure_random(32)
        random2 = security_service.generate_secure_random(32)
        
        assert random1 != random2
        assert len(random1) == 32
        assert len(random2) == 32
    
    def test_path_traversal_prevention(self, security_service):
        """Test path traversal attack prevention."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "uploads/../../../sensitive.txt",
            "files/../../config.json"
        ]
        
        for path in malicious_paths:
            sanitized = security_service.sanitize_file_path(path)
            assert ".." not in sanitized
            assert sanitized.startswith("uploads/") or sanitized == ""
    
    def test_header_injection_prevention(self, security_service):
        """Test HTTP header injection prevention."""
        malicious_headers = [
            "value\r\nX-Injected: malicious",
            "value\nSet-Cookie: session=hacked",
            "value\r\nLocation: http://evil.com"
        ]
        
        for header in malicious_headers:
            sanitized = security_service.sanitize_header_value(header)
            assert "\r" not in sanitized
            assert "\n" not in sanitized
    
    def test_command_injection_prevention(self, security_service):
        """Test command injection prevention."""
        malicious_inputs = [
            "file.txt; rm -rf /",
            "data | cat /etc/passwd",
            "input && malicious_command",
            "value`whoami`"
        ]
        
        for malicious_input in malicious_inputs:
            sanitized = security_service.sanitize_command_input(malicious_input)
            assert ";" not in sanitized
            assert "|" not in sanitized
            assert "&&" not in sanitized
            assert "`" not in sanitized


class TestInputValidation:
    """Test cases for input validation utilities."""
    
    def test_email_format_validation(self):
        """Test email format validation."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "first+last@test.org"
        ]
        
        invalid_emails = [
            "notanemail",
            "@example.com",
            "user@",
            "user @example.com",
            "user@domain",
            ""
        ]
        
        for email in valid_emails:
            assert validate_file_type(email, "email") is True
        
        for email in invalid_emails:
            assert validate_file_type(email, "email") is False
    
    def test_url_validation(self):
        """Test URL validation."""
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://sub.domain.com/page?param=value"
        ]
        
        invalid_urls = [
            "not a url",
            "javascript:alert('xss')",
            "file:///etc/passwd",
            "ftp://unsafe.com"
        ]
        
        for url in valid_urls:
            assert SecurityService.validate_url(url) is True
        
        for url in invalid_urls:
            assert SecurityService.validate_url(url) is False
    
    def test_integer_validation(self):
        """Test integer input validation."""
        assert SecurityService.validate_integer("123") is True
        assert SecurityService.validate_integer("-456") is True
        assert SecurityService.validate_integer("0") is True
        
        assert SecurityService.validate_integer("abc") is False
        assert SecurityService.validate_integer("12.34") is False
        assert SecurityService.validate_integer("") is False
    
    def test_uuid_validation(self):
        """Test UUID format validation."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        ]
        
        invalid_uuids = [
            "not-a-uuid",
            "12345",
            "550e8400-e29b-41d4-a716",
            ""
        ]
        
        for uuid in valid_uuids:
            assert SecurityService.validate_uuid(uuid) is True
        
        for uuid in invalid_uuids:
            assert SecurityService.validate_uuid(uuid) is False


class TestEncryption:
    """Test cases for encryption functionality."""
    
    @pytest.fixture
    def security_service(self):
        """Create security service instance."""
        return SecurityService()
    
    def test_symmetric_encryption_decryption(self, security_service):
        """Test symmetric encryption and decryption."""
        plaintext = "Sensitive data that needs protection"
        
        encrypted = security_service.encrypt_symmetric(plaintext)
        decrypted = security_service.decrypt_symmetric(encrypted)
        
        assert encrypted != plaintext
        assert decrypted == plaintext
    
    def test_encryption_with_different_keys(self, security_service):
        """Test that different keys produce different ciphertexts."""
        plaintext = "Test data"
        
        encrypted1 = security_service.encrypt_symmetric(plaintext, key="key1")
        encrypted2 = security_service.encrypt_symmetric(plaintext, key="key2")
        
        assert encrypted1 != encrypted2
    
    def test_encryption_of_empty_string(self, security_service):
        """Test encryption of empty string."""
        plaintext = ""
        
        encrypted = security_service.encrypt_symmetric(plaintext)
        decrypted = security_service.decrypt_symmetric(encrypted)
        
        assert decrypted == plaintext
    
    def test_encryption_of_large_content(self, security_service):
        """Test encryption of large content."""
        plaintext = "A" * 10000  # 10KB of data
        
        encrypted = security_service.encrypt_symmetric(plaintext)
        decrypted = security_service.decrypt_symmetric(encrypted)
        
        assert decrypted == plaintext
        assert len(encrypted) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
