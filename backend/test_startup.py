"""Quick test to check if backend can start."""
import sys
print(f"Python: {sys.version}")
print(f"Path: {sys.executable}")

try:
    print("\n1. Testing imports...")
    from fastapi import FastAPI
    print("✓ FastAPI imported")
    
    from core.config import settings
    print(f"✓ Settings loaded: {settings.PROJECT_NAME}")
    
    from api.v1.api import api_router
    print("✓ API router imported")
    
    print("\n2. Testing database connection...")
    from core.database import engine
    print(f"✓ Database engine created: {engine.url}")
    
    print("\n3. All imports successful!")
    print("\nYou can now start the server with:")
    print("uvicorn main:app --reload --host 0.0.0.0 --port 8000")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
