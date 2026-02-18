#!/usr/bin/env python3
"""
Script to deploy ProofOfAuthorship contract to Polygon mainnet.
"""
import asyncio
import json
import sys
from pathlib import Path
from web3 import Web3
from eth_account import Account
import os
from dotenv import load_dotenv


class ContractDeployer:
    """Deploy and verify smart contracts."""
    
    def __init__(self, rpc_url: str, private_key: str):
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.account = Account.from_key(private_key)
        
        # Add PoA middleware for Polygon
        try:
            from web3.middleware import geth_poa_middleware
        except ImportError:
            from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
    
    def load_contract_artifacts(self) -> tuple:
        """Load contract bytecode and ABI."""
        # In a real deployment, these would come from compiled Solidity
        # For now, we'll use a minimal contract structure
        
        # This is a simplified version - in production you'd compile the actual contract
        contract_abi = [
            {
                "inputs": [
                    {"name": "_contentHash", "type": "bytes32"},
                    {"name": "_stylometricHash", "type": "bytes32"},
                    {"name": "_ipfsHash", "type": "string"},
                    {"name": "_student", "type": "address"},
                    {"name": "_authorshipScore", "type": "uint8"},
                    {"name": "_aiProbability", "type": "uint8"},
                    {"name": "_institutionId", "type": "string"},
                    {"name": "_courseId", "type": "string"}
                ],
                "name": "mintProofToken",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function"
            },
            {
                "inputs": [{"name": "_tokenId", "type": "uint256"}],
                "name": "getSubmission",
                "outputs": [
                    {
                        "components": [
                            {"name": "contentHash", "type": "bytes32"},
                            {"name": "stylometricHash", "type": "bytes32"},
                            {"name": "ipfsHash", "type": "string"},
                            {"name": "student", "type": "address"},
                            {"name": "timestamp", "type": "uint256"},
                            {"name": "authorshipScore", "type": "uint8"},
                            {"name": "aiProbability", "type": "uint8"},
                            {"name": "verified", "type": "bool"},
                            {"name": "institutionId", "type": "string"},
                            {"name": "courseId", "type": "string"}
                        ],
                        "name": "",
                        "type": "tuple"
                    }
                ],
                "type": "function"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "tokenId", "type": "uint256"},
                    {"indexed": True, "name": "student", "type": "address"},
                    {"indexed": False, "name": "contentHash", "type": "bytes32"}
                ],
                "name": "SubmissionMinted",
                "type": "event"
            }
        ]
        
        # Placeholder bytecode - in production, compile the actual contract
        contract_bytecode = "0x608060405234801561001057600080fd5b50..."  # This would be the actual compiled bytecode
        
        return contract_abi, contract_bytecode
    
    async def deploy_contract(self) -> tuple:
        """Deploy the ProofOfAuthorship contract."""
        print("🚀 Deploying ProofOfAuthorship Contract")
        print("=" * 50)
        
        # Check account balance
        balance = self.w3.eth.get_balance(self.account.address)
        balance_eth = self.w3.from_wei(balance, 'ether')
        print(f"Deployer address: {self.account.address}")
        print(f"Account balance: {balance_eth} MATIC")
        
        if balance_eth < 0.1:
            print("❌ Insufficient balance for deployment. Need at least 0.1 MATIC")
            return None, None
        
        # Load contract artifacts
        try:
            contract_abi, contract_bytecode = self.load_contract_artifacts()
        except Exception as e:
            print(f"❌ Failed to load contract artifacts: {e}")
            print("💡 Make sure to compile the contract first:")
            print("   cd ../blockchain && npx hardhat compile")
            return None, None
        
        # Create contract instance
        contract = self.w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
        
        # Get current gas price
        gas_price = self.w3.eth.gas_price
        gas_price_gwei = self.w3.from_wei(gas_price, 'gwei')
        print(f"Current gas price: {gas_price_gwei} gwei")
        
        # Estimate gas for deployment
        try:
            gas_estimate = contract.constructor().estimate_gas()
            print(f"Estimated gas: {gas_estimate}")
            
            # Calculate deployment cost
            deployment_cost = gas_estimate * gas_price
            deployment_cost_eth = self.w3.from_wei(deployment_cost, 'ether')
            print(f"Estimated deployment cost: {deployment_cost_eth} MATIC")
            
        except Exception as e:
            print(f"❌ Gas estimation failed: {e}")
            return None, None
        
        # Build deployment transaction
        try:
            transaction = contract.constructor().build_transaction({
                'from': self.account.address,
                'nonce': self.w3.eth.get_transaction_count(self.account.address),
                'gas': gas_estimate,
                'gasPrice': gas_price,
                'chainId': 137  # Polygon mainnet
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.key)
            
            # Send transaction
            print("📤 Sending deployment transaction...")
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            print(f"Transaction hash: {tx_hash.hex()}")
            
            # Wait for confirmation
            print("⏳ Waiting for confirmation...")
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            
            if tx_receipt.status == 1:
                contract_address = tx_receipt.contractAddress
                print(f"✅ Contract deployed successfully!")
                print(f"Contract address: {contract_address}")
                print(f"Block number: {tx_receipt.blockNumber}")
                print(f"Gas used: {tx_receipt.gasUsed}")
                
                return contract_address, contract_abi
            else:
                print("❌ Contract deployment failed")
                return None, None
                
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return None, None
    
    def save_deployment_info(self, contract_address: str, contract_abi: list):
        """Save deployment information."""
        deployment_info = {
            "network": "polygon",
            "chainId": 137,
            "contractAddress": contract_address,
            "deployerAddress": self.account.address,
            "deployedAt": self.w3.eth.get_block('latest')['timestamp'],
            "abi": contract_abi
        }
        
        # Save to deployment file
        deployment_path = Path("../blockchain/deployments/polygon.json")
        deployment_path.parent.mkdir(exist_ok=True)
        
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"✅ Deployment info saved to: {deployment_path}")
        
        # Update environment file template
        env_update = f"""
# Update your .env file with these values:
CONTRACT_ADDRESS={contract_address}
PROOF_OF_AUTHORSHIP_CONTRACT_ADDRESS={contract_address}
"""
        print(env_update)


async def main():
    """Main deployment function."""
    print("🔗 Polygon Contract Deployment Tool")
    print("=" * 60)
    
    # Load environment
    load_dotenv("prod.env")  # Load production environment
    
    rpc_url = os.getenv("POLYGON_RPC_URL")
    private_key = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
    
    if not rpc_url or not private_key:
        print("❌ Missing required environment variables")
        print("Make sure POLYGON_RPC_URL and BLOCKCHAIN_PRIVATE_KEY are set")
        sys.exit(1)
    
    # Create deployer
    deployer = ContractDeployer(rpc_url, private_key)
    
    # Check connection
    if not deployer.w3.is_connected():
        print("❌ Failed to connect to Polygon network")
        sys.exit(1)
    
    print(f"✅ Connected to Polygon network")
    print(f"Chain ID: {deployer.w3.eth.chain_id}")
    print(f"Latest block: {deployer.w3.eth.block_number}")
    
    # Deploy contract
    contract_address, contract_abi = await deployer.deploy_contract()
    
    if contract_address:
        deployer.save_deployment_info(contract_address, contract_abi)
        
        print("\n🎉 Deployment completed successfully!")
        print("\n📋 Next steps:")
        print("1. Update your .env file with the contract address")
        print("2. Verify the contract on Polygonscan")
        print("3. Test the contract with a small transaction")
        print("4. Update your application configuration")
        
        # Verification instructions
        print(f"\n🔍 To verify on Polygonscan:")
        print(f"1. Go to https://polygonscan.com/address/{contract_address}")
        print("2. Click 'Contract' tab")
        print("3. Click 'Verify and Publish'")
        print("4. Upload your contract source code")
    else:
        print("\n❌ Deployment failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())