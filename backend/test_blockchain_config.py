#!/usr/bin/env python3
"""
Test script to verify blockchain configuration and connectivity.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Import after loading .env
settings = None
blockchain_service = None


async def test_blockchain_config():
    """Test blockchain configuration and connectivity."""
    print("🔗 Testing Blockchain Configuration")
    print("=" * 50)
    
    # Check configuration
    print(f"RPC URL: {settings.POLYGON_RPC_URL}")
    print(f"Contract Address: {settings.CONTRACT_ADDRESS}")
    print(f"Proof of Authorship Contract: {settings.PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS}")
    print(f"Private Key Configured: {'Yes' if settings.BLOCKCHAIN_PRIVATE_KEY else 'No'}")
    
    # Test Web3 connection
    try:
        if blockchain_service.w3.is_connected():
            print("✅ Web3 connection successful")
            
            # Get network info
            chain_id = blockchain_service.w3.eth.chain_id
            block_number = blockchain_service.w3.eth.block_number
            print(f"   Chain ID: {chain_id}")
            print(f"   Latest Block: {block_number}")
            
        else:
            print("❌ Web3 connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Web3 connection error: {e}")
        return False
    
    # Test contract initialization
    if blockchain_service.contract:
        print("✅ Smart contract initialized")
        print(f"   Contract Address: {blockchain_service.contract_address}")
    else:
        print("❌ Smart contract not initialized")
        print("   This is expected if the contract hasn't been deployed yet")
    
    # Test account configuration
    if blockchain_service.account:
        print("✅ Blockchain account configured")
        print(f"   Account Address: {blockchain_service.account.address}")
        
        # Check account balance
        try:
            balance = blockchain_service.w3.eth.get_balance(blockchain_service.account.address)
            balance_eth = blockchain_service.w3.from_wei(balance, 'ether')
            print(f"   Account Balance: {balance_eth} ETH")
        except Exception as e:
            print(f"   Could not fetch balance: {e}")
    else:
        print("❌ Blockchain account not configured")
        return False
    
    # Test network stats
    try:
        stats = await blockchain_service.get_network_stats()
        print("✅ Network stats retrieved")
        print(f"   Network: {stats.get('network', 'unknown')}")
        print(f"   Status: {stats.get('network_status', 'unknown')}")
        print(f"   Gas Price: {stats.get('gas_price', 'unknown')} gwei")
    except Exception as e:
        print(f"❌ Network stats error: {e}")
    
    print("\n🎉 Blockchain configuration test completed!")
    return True


async def test_ipfs_config():
    """Test IPFS configuration."""
    print("\n📁 Testing IPFS Configuration")
    print("=" * 50)
    
    print(f"IPFS API URL: {settings.IPFS_API_URL}")
    print(f"IPFS Gateway URL: {settings.IPFS_GATEWAY_URL}")
    
    try:
        from services.ipfs_service import ipfs_service
        
        # Test IPFS connection
        status = await ipfs_service.check_connection()
        if status:
            print("✅ IPFS connection successful")
        else:
            print("❌ IPFS connection failed")
            print("   Make sure IPFS node is running on the configured URL")
            
    except Exception as e:
        print(f"❌ IPFS test error: {e}")


def main():
    """Main test function."""
    print("🚀 Stylos Blockchain Configuration Test")
    print("=" * 60)
    
    # Load environment variables from the correct path
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    print(f"Loading .env from: {env_path}")
    print(f".env exists: {env_path.exists()}")
    
    # Import settings after loading .env
    global settings, blockchain_service
    from core.config import settings
    from services.blockchain_service import blockchain_service
    
    # Run tests
    try:
        asyncio.run(test_blockchain_config())
        asyncio.run(test_ipfs_config())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()