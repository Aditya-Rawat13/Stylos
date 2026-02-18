#!/usr/bin/env python3
"""
Script to generate secure production keys and configuration.
"""
import secrets
import os
from pathlib import Path
from eth_account import Account
from cryptography.fernet import Fernet
import base64
import hashlib


def generate_secure_private_key():
    """Generate a cryptographically secure private key."""
    # Generate 32 random bytes for private key
    private_key_bytes = secrets.token_bytes(32)
    private_key_hex = private_key_bytes.hex()
    
    # Create account from private key
    account = Account.from_key(private_key_hex)
    
    return f"0x{private_key_hex}", account.address


def generate_encryption_key():
    """Generate a secure encryption key."""
    return Fernet.generate_key().decode()


def generate_jwt_secret():
    """Generate a secure JWT secret."""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


def generate_webhook_secret():
    """Generate a secure webhook secret."""
    return base64.urlsafe_b64encode(secrets.token_bytes(24)).decode()


def create_production_env():
    """Create production environment configuration."""
    print("🔐 Generating Production Keys and Configuration")
    print("=" * 60)
    
    # Generate blockchain keys
    private_key, address = generate_secure_private_key()
    print(f"✅ Generated blockchain private key")
    print(f"   Address: {address}")
    print(f"   ⚠️  IMPORTANT: Fund this address with MATIC tokens!")
    
    # Generate other secrets
    encryption_key = generate_encryption_key()
    jwt_secret = generate_jwt_secret()
    app_secret = generate_jwt_secret()  # For general app encryption
    webhook_secret = generate_webhook_secret()
    ipfs_encryption_password = generate_jwt_secret()
    
    print(f"✅ Generated application secrets")
    
    # Create production .env template
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    prod_env_content = f"""# Production Environment Configuration
# Generated on: {current_time}
# 
# ⚠️  SECURITY WARNING: Keep this file secure and never commit to version control!

# Application Configuration
DEBUG=false
SECRET_KEY={app_secret}
JWT_SECRET_KEY={jwt_secret}

# Database Configuration (UPDATE WITH YOUR PRODUCTION DATABASE)
DATABASE_URL=postgresql://stylos_user:CHANGE_THIS_PASSWORD@your-db-host:5432/stylos_prod

# Redis Configuration (UPDATE WITH YOUR PRODUCTION REDIS)
REDIS_URL=redis://your-redis-host:6379/0

# Blockchain Configuration - PRODUCTION
POLYGON_RPC_URL=https://polygon-rpc.com
# Alternative RPC endpoints:
# POLYGON_RPC_URL=https://rpc-mainnet.maticvigil.com
# POLYGON_RPC_URL=https://polygon-mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID

# ⚠️  DEPLOY YOUR CONTRACT FIRST, THEN UPDATE THESE VALUES
CONTRACT_ADDRESS=DEPLOY_CONTRACT_FIRST
PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS=DEPLOY_CONTRACT_FIRST
BLOCKCHAIN_PRIVATE_KEY={private_key}

# IPFS Configuration - PRODUCTION
# Option 1: Hosted IPFS (Recommended)
IPFS_API_URL=https://ipfs.infura.io:5001
IPFS_API_KEY=YOUR_INFURA_PROJECT_ID
IPFS_API_SECRET=YOUR_INFURA_PROJECT_SECRET
IPFS_GATEWAY_URL=https://ipfs.io/ipfs/

# Option 2: Self-hosted IPFS
# IPFS_API_URL=https://your-ipfs-node:5001
# IPFS_GATEWAY_URL=https://your-ipfs-gateway/ipfs/

IPFS_ENCRYPTION_PASSWORD={ipfs_encryption_password}

# File Upload Configuration
MAX_FILE_SIZE=10485760
UPLOAD_DIR=/var/app/uploads

# ML Model Configuration
MODEL_CACHE_DIR=/var/app/models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# CORS Configuration (UPDATE WITH YOUR DOMAIN)
ALLOWED_HOSTS=["https://your-domain.com","https://www.your-domain.com"]

# Security Configuration
ENCRYPTION_MASTER_KEY={encryption_key}
ENABLE_DATA_ENCRYPTION=true
ENABLE_INTRUSION_DETECTION=true
SECURITY_MONITORING_ENABLED=true
FORCE_HTTPS=true
SECURE_COOKIES=true

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST_SIZE=20

# Session Security
SESSION_TIMEOUT_MINUTES=60

# Compliance
GDPR_COMPLIANCE_ENABLED=true
FERPA_COMPLIANCE_ENABLED=true
DATA_RETENTION_ENABLED=true

# Audit Logging
AUDIT_LOG_RETENTION_DAYS=365
AUDIT_LOG_ENCRYPTION=true

# Email Configuration (UPDATE WITH YOUR SMTP SETTINGS)
SMTP_SERVER=smtp.your-provider.com
SMTP_PORT=587
SMTP_USERNAME=your-email@your-domain.com
SMTP_PASSWORD=YOUR_EMAIL_PASSWORD
SMTP_USE_TLS=true
FROM_EMAIL=noreply@your-domain.com
FRONTEND_URL=https://your-domain.com

# LMS Integration Configuration
LMS_INTEGRATION_ENABLED=true
LMS_WEBHOOK_BASE_URL=https://your-domain.com/api/v1/lms/webhook
LMS_WEBHOOK_SECRET={webhook_secret}
"""
    
    # Write to production env file
    prod_env_path = Path("prod.env")
    with open(prod_env_path, 'w', encoding='utf-8') as f:
        f.write(prod_env_content)
    
    print(f"✅ Created production environment file: {prod_env_path}")
    
    # Create deployment checklist
    checklist_content = f"""# Production Deployment Checklist

## 🔐 Security Setup
- [ ] Generated new private key: {address}
- [ ] Fund blockchain account with MATIC tokens (minimum 1 MATIC recommended)
- [ ] Update database credentials in prod.env
- [ ] Update Redis credentials in prod.env
- [ ] Configure SMTP settings for email notifications
- [ ] Set up SSL certificates for HTTPS
- [ ] Configure firewall rules

## 🚀 Blockchain Deployment
- [ ] Deploy ProofOfAuthorship contract to Polygon mainnet
- [ ] Update CONTRACT_ADDRESS in prod.env
- [ ] Update PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS in prod.env
- [ ] Test contract deployment with small transaction
- [ ] Verify contract on Polygonscan

## 📁 IPFS Setup
Choose one option:

### Option A: Hosted IPFS (Recommended)
- [ ] Create Infura IPFS project
- [ ] Update IPFS_API_KEY in prod.env
- [ ] Update IPFS_API_SECRET in prod.env
- [ ] Test IPFS connectivity

### Option B: Self-hosted IPFS
- [ ] Set up IPFS node on production server
- [ ] Configure IPFS API endpoint
- [ ] Set up IPFS gateway
- [ ] Configure backup/redundancy

## 🌐 Domain and Infrastructure
- [ ] Configure domain DNS
- [ ] Set up load balancer
- [ ] Configure CDN (optional)
- [ ] Set up monitoring and logging
- [ ] Configure backup systems

## 🧪 Testing
- [ ] Run configuration test: `python test_blockchain_config.py`
- [ ] Test blockchain transactions
- [ ] Test IPFS storage and retrieval
- [ ] Test end-to-end submission flow
- [ ] Load testing
- [ ] Security testing

## 📊 Monitoring
- [ ] Set up application monitoring
- [ ] Configure blockchain monitoring
- [ ] Set up alerting for low gas balance
- [ ] Monitor IPFS storage usage
- [ ] Set up log aggregation

## 🔒 Final Security Checks
- [ ] Verify all secrets are unique and secure
- [ ] Ensure .env files are not in version control
- [ ] Set up secret management system
- [ ] Configure access controls
- [ ] Document incident response procedures

## 💰 Cost Considerations
- [ ] Estimate gas costs for expected transaction volume
- [ ] Set up gas price monitoring
- [ ] Configure automatic balance alerts
- [ ] Plan for IPFS storage costs
- [ ] Budget for infrastructure costs
"""
    
    checklist_path = Path("PRODUCTION_DEPLOYMENT_CHECKLIST.md")
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist_content)
    
    print(f"✅ Created deployment checklist: {checklist_path}")
    
    # Security warnings
    print("\n🚨 CRITICAL SECURITY WARNINGS:")
    print("1. NEVER commit prod.env to version control")
    print("2. Store secrets in a secure secret management system")
    print("3. Fund the blockchain account before deployment")
    print("4. Test everything on testnet first")
    print("5. Set up monitoring and alerting")
    print(f"6. Blockchain address to fund: {address}")
    
    return private_key, address


if __name__ == "__main__":
    create_production_env()