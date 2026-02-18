"""
Blockchain and IPFS configuration settings.
"""
import os
from typing import List, Optional


class BlockchainConfig:
    """Blockchain configuration settings."""
    
    # Network Configuration
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "http://127.0.0.1:8545")
    MUMBAI_RPC_URL: str = os.getenv("MUMBAI_RPC_URL", "http://127.0.0.1:8545")
    ETHEREUM_RPC_URL: str = os.getenv("ETHEREUM_RPC_URL", "http://127.0.0.1:8545")
    
    # Contract Configuration
    PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS: str = os.getenv("PROOF_OF_AUTHORSHIP_PROXY", "")
    CONTRACT_ABI_PATH: str = os.getenv("CONTRACT_ABI_PATH", "blockchain/deployments/polygon.json")
    
    # Account Configuration
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
    MNEMONIC: Optional[str] = os.getenv("MNEMONIC")
    
    # Gas Configuration
    DEFAULT_GAS_PRICE: int = int(os.getenv("DEFAULT_GAS_PRICE", "30000000000"))  # 30 gwei
    MAX_GAS_PRICE: int = int(os.getenv("MAX_GAS_PRICE", "100000000000"))  # 100 gwei
    GAS_LIMIT: int = int(os.getenv("GAS_LIMIT", "8000000"))
    
    # Network Settings
    NETWORK_ID: int = int(os.getenv("NETWORK_ID", "31337"))  # Hardhat local network
    NETWORK_NAME: str = os.getenv("NETWORK_NAME", "localhost")
    CHAIN_ID: int = int(os.getenv("CHAIN_ID", "31337"))
    CONFIRMATION_BLOCKS: int = int(os.getenv("CONFIRMATION_BLOCKS", "1"))
    
    # Retry Configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY: int = int(os.getenv("RETRY_DELAY", "60"))  # seconds
    
    # Monitoring
    ENABLE_EVENT_MONITORING: bool = os.getenv("ENABLE_EVENT_MONITORING", "true").lower() == "true"
    EVENT_POLLING_INTERVAL: int = int(os.getenv("EVENT_POLLING_INTERVAL", "10"))  # seconds


class IPFSConfig:
    """IPFS configuration settings."""
    
    # IPFS Node Configuration
    IPFS_API_URL: str = os.getenv("IPFS_API_URL", "https://ipfs.infura.io:5001")
    IPFS_GATEWAY_URL: str = os.getenv("IPFS_GATEWAY_URL", "https://ipfs.io/ipfs")
    
    # Authentication
    IPFS_API_KEY: Optional[str] = os.getenv("IPFS_API_KEY")
    IPFS_API_SECRET: Optional[str] = os.getenv("IPFS_API_SECRET")
    
    # Encryption
    IPFS_ENCRYPTION_PASSWORD: str = os.getenv("IPFS_ENCRYPTION_PASSWORD", "stylos-default-key")
    ENABLE_ENCRYPTION: bool = os.getenv("ENABLE_ENCRYPTION", "true").lower() == "true"
    
    # Redundancy and Pinning
    IPFS_REDUNDANCY_NODES: List[str] = os.getenv("IPFS_REDUNDANCY_NODES", "").split(",") if os.getenv("IPFS_REDUNDANCY_NODES") else []
    AUTO_PIN_CONTENT: bool = os.getenv("AUTO_PIN_CONTENT", "true").lower() == "true"
    PIN_EXPIRY_DAYS: int = int(os.getenv("PIN_EXPIRY_DAYS", "365"))
    
    # Performance
    IPFS_TIMEOUT: int = int(os.getenv("IPFS_TIMEOUT", "30"))  # seconds
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
    
    # Content Management
    ENABLE_CONTENT_CLEANUP: bool = os.getenv("ENABLE_CONTENT_CLEANUP", "false").lower() == "true"
    CLEANUP_INTERVAL_HOURS: int = int(os.getenv("CLEANUP_INTERVAL_HOURS", "24"))


# Global configuration instances
blockchain_config = BlockchainConfig()
ipfs_config = IPFSConfig()