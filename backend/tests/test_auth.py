"""
Tests for authentication system.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.security_service import security_service


async def test_password_strength_validation():
    """Test password strength validation."""
    # Test weak password
    weak_result = await security_service.validate_password_strength("123")
    assert not weak_result["is_valid"]
    assert weak_result["score"] < 50
    
    # Test strong password
    strong_result = await security_service.validate_password_strength("StrongPass123!")
    assert strong_result["is_valid"]
    assert strong_result["score"] >= 80
    
    return True

def test_jwt_token_creation():
    """Test JWT token creation and validation."""
    from core.security import security
    
    # Test access token creation
    token_data = {"sub": "123"}
    access_token = security.create_access_token(token_data)
    assert access_token is not None
    assert isinstance(access_token, str)
    
    # Test token verification
    payload = security.verify_token(access_token, "access")
    assert payload["sub"] == "123"
    assert payload["type"] == "access"
    
    return True

def test_password_hashing():
    """Test password hashing and verification."""
    from core.security import security
    
    password = "TestPassword123!"
    hashed = security.hash_password(password)
    
    # Verify correct password
    assert security.verify_password(password, hashed)
    
    # Verify incorrect password
    assert not security.verify_password("WrongPassword", hashed)
    
    return True


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        print("Testing password strength validation...")
        await test_password_strength_validation()
        print("✓ Password strength validation works")
        
        print("Testing JWT token creation...")
        test_jwt_token_creation()
        print("✓ JWT token creation works")
        
        print("Testing password hashing...")
        test_password_hashing()
        print("✓ Password hashing works")
        
        print("\nAll authentication system tests passed! ✓")
    
    # Run tests
    asyncio.run(run_tests())