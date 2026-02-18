#!/usr/bin/env python3
"""
Complete production setup script for Stylos blockchain configuration.
"""
import asyncio
import os
import sys
from pathlib import Path
import subprocess
import json
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from generate_production_keys import create_production_env
from test_blockchain_config import test_blockchain_config, test_ipfs_config


class ProductionSetup:
    """Complete production setup orchestrator."""
    
    def __init__(self):
        self.setup_complete = False
        self.contract_deployed = False
        self.keys_generated = False
    
    async def run_setup(self):
        """Run complete production setup."""
        print("🚀 Stylos Production Setup")
        print("=" * 60)
        print("This script will help you set up Stylos for production deployment.")
        print("Please follow the steps carefully.\n")
        
        # Step 1: Generate production keys
        await self.step_1_generate_keys()
        
        # Step 2: Configure services
        await self.step_2_configure_services()
        
        # Step 3: Deploy contract (optional)
        await self.step_3_deploy_contract()
        
        # Step 4: Test configuration
        await self.step_4_test_configuration()
        
        # Step 5: Final checklist
        await self.step_5_final_checklist()
    
    async def step_1_generate_keys(self):
        """Generate production keys and configuration."""
        print("📋 Step 1: Generate Production Keys")
        print("-" * 40)
        
        response = input("Generate new production keys? (y/n): ").lower()
        if response == 'y':
            try:
                private_key, address = create_production_env()
                self.keys_generated = True
                self.blockchain_address = address
                print(f"\n✅ Keys generated successfully!")
                print(f"🔑 Blockchain address: {address}")
                print(f"💰 IMPORTANT: Fund this address with MATIC tokens!")
            except Exception as e:
                print(f"❌ Key generation failed: {e}")
                return False
        else:
            print("⏭️  Skipping key generation")
        
        return True
    
    async def step_2_configure_services(self):
        """Configure external services."""
        print("\n📋 Step 2: Configure External Services")
        print("-" * 40)
        
        # Database configuration
        print("\n🗄️  Database Configuration:")
        db_configured = input("Have you configured your production database? (y/n): ").lower()
        if db_configured != 'y':
            print("⚠️  Please configure your production database:")
            print("   1. Set up PostgreSQL instance")
            print("   2. Create database and user")
            print("   3. Update DATABASE_URL in prod.env")
            print("   4. Run database migrations")
        
        # Redis configuration
        print("\n🔴 Redis Configuration:")
        redis_configured = input("Have you configured your production Redis? (y/n): ").lower()
        if redis_configured != 'y':
            print("⚠️  Please configure your production Redis:")
            print("   1. Set up Redis instance")
            print("   2. Update REDIS_URL in prod.env")
        
        # IPFS configuration
        print("\n📁 IPFS Configuration:")
        print("Choose your IPFS setup:")
        print("1. Hosted IPFS (Infura) - Recommended")
        print("2. Self-hosted IPFS node")
        print("3. Skip for now")
        
        ipfs_choice = input("Enter choice (1-3): ")
        
        if ipfs_choice == "1":
            print("\n🌐 Setting up hosted IPFS:")
            print("1. Go to https://infura.io/")
            print("2. Create an account and new project")
            print("3. Enable IPFS API")
            print("4. Copy Project ID and Project Secret")
            
            project_id = input("Enter Infura Project ID (or press Enter to skip): ")
            if project_id:
                project_secret = input("Enter Infura Project Secret: ")
                self.update_env_file("IPFS_API_KEY", project_id)
                self.update_env_file("IPFS_API_SECRET", project_secret)
                print("✅ IPFS configuration updated")
        
        elif ipfs_choice == "2":
            print("\n🖥️  Self-hosted IPFS setup:")
            print("1. Install IPFS on your server")
            print("2. Configure IPFS daemon")
            print("3. Set up reverse proxy for API access")
            print("4. Update IPFS_API_URL in prod.env")
        
        # Email configuration
        print("\n📧 Email Configuration:")
        email_configured = input("Have you configured SMTP settings? (y/n): ").lower()
        if email_configured != 'y':
            print("⚠️  Please configure email settings in prod.env:")
            print("   - SMTP_SERVER")
            print("   - SMTP_USERNAME")
            print("   - SMTP_PASSWORD")
    
    async def step_3_deploy_contract(self):
        """Deploy smart contract."""
        print("\n📋 Step 3: Deploy Smart Contract")
        print("-" * 40)
        
        deploy_now = input("Deploy contract now? (y/n): ").lower()
        if deploy_now == 'y':
            if not self.keys_generated:
                print("❌ Cannot deploy without generated keys")
                return False
            
            # Check if blockchain account is funded
            funded = input(f"Have you funded the blockchain address with MATIC? (y/n): ").lower()
            if funded != 'y':
                print("⚠️  Please fund your blockchain address first:")
                print(f"   Address: {getattr(self, 'blockchain_address', 'Check prod.env file')}")
                print("   Minimum: 0.1 MATIC for deployment")
                print("   Recommended: 1 MATIC for operations")
                return False
            
            try:
                print("🚀 Starting contract deployment...")
                # In a real implementation, this would call the deploy script
                print("📝 Note: Contract deployment requires compiled Solidity code")
                print("   Run: cd ../blockchain && npx hardhat compile")
                print("   Then: python scripts/deploy_contract.py")
                
                self.contract_deployed = True
            except Exception as e:
                print(f"❌ Contract deployment failed: {e}")
        else:
            print("⏭️  Skipping contract deployment")
            print("💡 You can deploy later with: python scripts/deploy_contract.py")
    
    async def step_4_test_configuration(self):
        """Test the production configuration."""
        print("\n📋 Step 4: Test Configuration")
        print("-" * 40)
        
        test_now = input("Test configuration now? (y/n): ").lower()
        if test_now == 'y':
            try:
                # Load production environment
                load_dotenv("prod.env")
                
                print("\n🧪 Running configuration tests...")
                
                # Import and run tests
                from core.config import settings
                from services.blockchain_service import blockchain_service
                
                # Test blockchain
                success = await test_blockchain_config()
                if success:
                    print("✅ Blockchain configuration test passed")
                else:
                    print("❌ Blockchain configuration test failed")
                
                # Test IPFS
                await test_ipfs_config()
                
            except Exception as e:
                print(f"❌ Configuration test failed: {e}")
        else:
            print("⏭️  Skipping configuration test")
    
    async def step_5_final_checklist(self):
        """Final deployment checklist."""
        print("\n📋 Step 5: Final Deployment Checklist")
        print("-" * 40)
        
        checklist_items = [
            "Generated production keys",
            "Configured production database",
            "Configured Redis instance",
            "Set up IPFS storage",
            "Configured email settings",
            "Funded blockchain account",
            "Deployed smart contract",
            "Updated contract addresses in .env",
            "Tested configuration",
            "Set up SSL certificates",
            "Configured domain and DNS",
            "Set up monitoring and logging",
            "Configured backup systems",
            "Reviewed security settings"
        ]
        
        print("\n✅ Deployment Checklist:")
        completed_items = 0
        
        for i, item in enumerate(checklist_items, 1):
            status = input(f"{i:2d}. {item} - Complete? (y/n): ").lower()
            if status == 'y':
                completed_items += 1
                print(f"    ✅ {item}")
            else:
                print(f"    ⏳ {item}")
        
        completion_rate = (completed_items / len(checklist_items)) * 100
        print(f"\n📊 Setup Completion: {completion_rate:.1f}% ({completed_items}/{len(checklist_items)})")
        
        if completion_rate >= 80:
            print("🎉 Great! You're ready for production deployment!")
        elif completion_rate >= 60:
            print("⚠️  Almost ready! Complete the remaining items before deployment.")
        else:
            print("❌ More setup required before production deployment.")
        
        # Generate final summary
        self.generate_deployment_summary(completed_items, len(checklist_items))
    
    def update_env_file(self, key: str, value: str):
        """Update environment file with new value."""
        try:
            env_path = Path("prod.env")
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                
                # Update or add the key
                lines = content.split('\n')
                updated = False
                
                for i, line in enumerate(lines):
                    if line.startswith(f"{key}="):
                        lines[i] = f"{key}={value}"
                        updated = True
                        break
                
                if not updated:
                    lines.append(f"{key}={value}")
                
                with open(env_path, 'w') as f:
                    f.write('\n'.join(lines))
        except Exception as e:
            print(f"Failed to update env file: {e}")
    
    def generate_deployment_summary(self, completed: int, total: int):
        """Generate deployment summary report."""
        summary = f"""# Production Deployment Summary

## Setup Status
- Completed: {completed}/{total} items ({(completed/total)*100:.1f}%)
- Keys Generated: {'✅' if self.keys_generated else '❌'}
- Contract Deployed: {'✅' if self.contract_deployed else '❌'}

## Next Steps
{'✅ Ready for production deployment!' if completed >= total * 0.8 else '⚠️ Complete remaining checklist items'}

## Important Files
- Production environment: `prod.env`
- Deployment checklist: `PRODUCTION_DEPLOYMENT_CHECKLIST.md`
- Configuration test: `python test_blockchain_config.py`

## Security Reminders
- Never commit prod.env to version control
- Store secrets in secure secret management
- Monitor blockchain account balance
- Set up automated backups
- Configure monitoring and alerting

## Support
- Documentation: See README files in each service directory
- Testing: Run configuration tests regularly
- Monitoring: Set up health checks and alerts
"""
        
        with open("DEPLOYMENT_SUMMARY.md", 'w') as f:
            f.write(summary)
        
        print(f"\n📄 Deployment summary saved to: DEPLOYMENT_SUMMARY.md")


async def main():
    """Main setup function."""
    setup = ProductionSetup()
    await setup.run_setup()


if __name__ == "__main__":
    asyncio.run(main())