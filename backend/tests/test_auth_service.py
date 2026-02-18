"""
Unit tests for authentication service.
Tests user authentication, authorization, and token management.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from services.auth_service import AuthService
from core.auth import create_access_token, verify_password, get_password_hash
from utils.exceptions import AuthenticationError, AuthorizationError


class TestAuthService:
    """Test cases for AuthService."""
    
    @pytest.fixture
    def auth_service(self):
        """Create auth service instance."""
        return AuthService()
    
    def test_password_hashing(self):
        """Test password hashing functionality."""
        password = "SecurePassword123!"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > 20
        assert verify_password(password, hashed)
    
    def test_password_verification_success(self):
        """Test successful password verification."""
        password = "TestPassword123!"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test failed password verification."""
        password = "CorrectPassword123!"
        wrong_password = "WrongPassword123!"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_access_token_creation(self):
        """Test JWT access token creation."""
        data = {"sub": "user123", "role": "student"}
        token = create_access_token(data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 50
    
    def test_access_token_with_expiration(self):
        """Test access token with custom expiration."""
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta=expires_delta)
        
        assert token is not None
        assert isinstance(token, str)
    
    @pytest.mark.asyncio
    async def test_user_registration(self, auth_service):
        """Test user registration process."""
        with patch.object(auth_service, '_create_user_in_db', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = {
                "id": "user123",
                "email": "newuser@test.com",
                "name": "New User",
                "role": "student"
            }
            
            result = await auth_service.register_user(
                email="newuser@test.com",
                password="SecurePass123!",
                name="New User"
            )
            
            assert result["email"] == "newuser@test.com"
            assert result["name"] == "New User"
            assert "password" not in result
    
    @pytest.mark.asyncio
    async def test_user_login_success(self, auth_service):
        """Test successful user login."""
        with patch.object(auth_service, '_get_user_by_email', new_callable=AsyncMock) as mock_get:
            password = "TestPassword123!"
            hashed = get_password_hash(password)
            
            mock_get.return_value = {
                "id": "user123",
                "email": "test@test.com",
                "password_hash": hashed,
                "role": "student"
            }
            
            result = await auth_service.authenticate_user("test@test.com", password)
            
            assert result is not None
            assert result["email"] == "test@test.com"
    
    @pytest.mark.asyncio
    async def test_user_login_wrong_password(self, auth_service):
        """Test login with wrong password."""
        with patch.object(auth_service, '_get_user_by_email', new_callable=AsyncMock) as mock_get:
            correct_password = "CorrectPassword123!"
            wrong_password = "WrongPassword123!"
            hashed = get_password_hash(correct_password)
            
            mock_get.return_value = {
                "id": "user123",
                "email": "test@test.com",
                "password_hash": hashed,
                "role": "student"
            }
            
            result = await auth_service.authenticate_user("test@test.com", wrong_password)
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_user_login_nonexistent_user(self, auth_service):
        """Test login with nonexistent user."""
        with patch.object(auth_service, '_get_user_by_email', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            
            result = await auth_service.authenticate_user("nonexistent@test.com", "password")
            
            assert result is None
    
    def test_role_based_authorization_student(self):
        """Test role-based authorization for student."""
        user_data = {"id": "user123", "role": "student"}
        
        # Students can access student endpoints
        assert auth_service._check_permission(user_data, "student") is True
        
        # Students cannot access admin endpoints
        assert auth_service._check_permission(user_data, "admin") is False
    
    def test_role_based_authorization_admin(self):
        """Test role-based authorization for admin."""
        user_data = {"id": "admin123", "role": "admin"}
        
        # Admins can access admin endpoints
        assert auth_service._check_permission(user_data, "admin") is True
        
        # Admins can also access student endpoints
        assert auth_service._check_permission(user_data, "student") is True
    
    @pytest.mark.asyncio
    async def test_token_refresh(self, auth_service):
        """Test token refresh functionality."""
        with patch.object(auth_service, '_validate_refresh_token', new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = {
                "id": "user123",
                "email": "test@test.com",
                "role": "student"
            }
            
            new_token = await auth_service.refresh_access_token("old_refresh_token")
            
            assert new_token is not None
            assert isinstance(new_token, str)
    
    @pytest.mark.asyncio
    async def test_password_reset_request(self, auth_service):
        """Test password reset request."""
        with patch.object(auth_service, '_get_user_by_email', new_callable=AsyncMock) as mock_get:
            with patch.object(auth_service, '_send_reset_email', new_callable=AsyncMock) as mock_send:
                mock_get.return_value = {
                    "id": "user123",
                    "email": "test@test.com"
                }
                
                result = await auth_service.request_password_reset("test@test.com")
                
                assert result is True
                mock_send.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_password_reset_completion(self, auth_service):
        """Test password reset completion."""
        with patch.object(auth_service, '_validate_reset_token', new_callable=AsyncMock) as mock_validate:
            with patch.object(auth_service, '_update_password', new_callable=AsyncMock) as mock_update:
                mock_validate.return_value = {"id": "user123"}
                mock_update.return_value = True
                
                result = await auth_service.reset_password("reset_token", "NewPassword123!")
                
                assert result is True
                mock_update.assert_called_once()
    
    def test_password_strength_validation_weak(self):
        """Test password strength validation for weak passwords."""
        weak_passwords = [
            "short",
            "12345678",
            "password",
            "abcdefgh"
        ]
        
        for password in weak_passwords:
            assert auth_service._validate_password_strength(password) is False
    
    def test_password_strength_validation_strong(self):
        """Test password strength validation for strong passwords."""
        strong_passwords = [
            "SecurePass123!",
            "MyP@ssw0rd2024",
            "C0mpl3x!Pass"
        ]
        
        for password in strong_passwords:
            assert auth_service._validate_password_strength(password) is True
    
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
            "user @example.com"
        ]
        
        for email in valid_emails:
            assert auth_service._validate_email_format(email) is True
        
        for email in invalid_emails:
            assert auth_service._validate_email_format(email) is False
    
    @pytest.mark.asyncio
    async def test_duplicate_email_registration(self, auth_service):
        """Test registration with duplicate email."""
        with patch.object(auth_service, '_get_user_by_email', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "id": "existing_user",
                "email": "existing@test.com"
            }
            
            with pytest.raises(Exception):
                await auth_service.register_user(
                    email="existing@test.com",
                    password="Password123!",
                    name="New User"
                )
    
    @pytest.mark.asyncio
    async def test_session_management(self, auth_service):
        """Test session creation and validation."""
        with patch.object(auth_service, '_create_session', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "session_id_123"
            
            session_id = await auth_service.create_session("user123")
            
            assert session_id is not None
            assert isinstance(session_id, str)
    
    @pytest.mark.asyncio
    async def test_session_invalidation(self, auth_service):
        """Test session invalidation on logout."""
        with patch.object(auth_service, '_invalidate_session', new_callable=AsyncMock) as mock_invalidate:
            mock_invalidate.return_value = True
            
            result = await auth_service.logout("session_id_123")
            
            assert result is True
            mock_invalidate.assert_called_once()
    
    def test_token_expiration_check(self):
        """Test token expiration validation."""
        # Create expired token
        expired_data = {"sub": "user123", "exp": datetime.utcnow() - timedelta(hours=1)}
        
        # Create valid token
        valid_data = {"sub": "user123", "exp": datetime.utcnow() + timedelta(hours=1)}
        
        assert auth_service._is_token_expired(expired_data) is True
        assert auth_service._is_token_expired(valid_data) is False
    
    @pytest.mark.asyncio
    async def test_account_lockout_after_failed_attempts(self, auth_service):
        """Test account lockout after multiple failed login attempts."""
        with patch.object(auth_service, '_get_failed_attempts', new_callable=AsyncMock) as mock_get_attempts:
            with patch.object(auth_service, '_lock_account', new_callable=AsyncMock) as mock_lock:
                mock_get_attempts.return_value = 5  # Max attempts reached
                
                result = await auth_service._check_account_status("user123")
                
                assert result["locked"] is True
    
    @pytest.mark.asyncio
    async def test_email_verification(self, auth_service):
        """Test email verification process."""
        with patch.object(auth_service, '_validate_verification_token', new_callable=AsyncMock) as mock_validate:
            with patch.object(auth_service, '_mark_email_verified', new_callable=AsyncMock) as mock_mark:
                mock_validate.return_value = {"user_id": "user123"}
                mock_mark.return_value = True
                
                result = await auth_service.verify_email("verification_token")
                
                assert result is True
                mock_mark.assert_called_once()
    
    def test_permission_hierarchy(self):
        """Test permission hierarchy (admin > student)."""
        admin_user = {"id": "admin1", "role": "admin"}
        student_user = {"id": "student1", "role": "student"}
        
        # Admin has all permissions
        assert auth_service._has_permission(admin_user, "view_all_submissions") is True
        assert auth_service._has_permission(admin_user, "view_own_submissions") is True
        
        # Student has limited permissions
        assert auth_service._has_permission(student_user, "view_all_submissions") is False
        assert auth_service._has_permission(student_user, "view_own_submissions") is True


class TestTokenSecurity:
    """Test cases for token security."""
    
    def test_token_contains_no_sensitive_data(self):
        """Test that tokens don't contain sensitive data."""
        data = {"sub": "user123", "role": "student"}
        token = create_access_token(data)
        
        # Token should not contain password or other sensitive info
        assert "password" not in token.lower()
        assert "secret" not in token.lower()
    
    def test_token_signature_validation(self):
        """Test token signature validation."""
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        # Valid token should have 3 parts (header.payload.signature)
        parts = token.split(".")
        assert len(parts) == 3
    
    def test_different_tokens_for_same_user(self):
        """Test that different tokens are generated for same user."""
        data = {"sub": "user123"}
        token1 = create_access_token(data)
        token2 = create_access_token(data)
        
        # Tokens should be different due to timestamp
        assert token1 != token2


class TestAuthorizationChecks:
    """Test cases for authorization checks."""
    
    def test_resource_ownership_check(self):
        """Test resource ownership authorization."""
        user = {"id": "user123", "role": "student"}
        resource = {"owner_id": "user123"}
        
        assert AuthService._check_resource_ownership(user, resource) is True
    
    def test_resource_ownership_check_failure(self):
        """Test resource ownership check failure."""
        user = {"id": "user123", "role": "student"}
        resource = {"owner_id": "different_user"}
        
        assert AuthService._check_resource_ownership(user, resource) is False
    
    def test_admin_bypass_ownership_check(self):
        """Test that admins can bypass ownership checks."""
        admin = {"id": "admin123", "role": "admin"}
        resource = {"owner_id": "different_user"}
        
        assert AuthService._check_resource_access(admin, resource) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
