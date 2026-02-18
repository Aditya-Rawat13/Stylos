#!/usr/bin/env python3
"""
Initialize security system with default configurations.
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy.ext.asyncio import AsyncSession
from core.database import get_db, engine
from services.key_management import key_management_service
from services.compliance_service import compliance_service
from services.security_service import security_service
from utils.encryption import get_encryption_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_security_system():
    """Initialize the security system with default configurations."""
    
    logger.info("Starting security system initialization...")
    
    try:
        async with get_db() as db:
            # 1. Initialize default data retention policies
            logger.info("Initializing default data retention policies...")
            await compliance_service.initialize_default_policies(db)
            
            # 2. Generate initial master encryption key
            logger.info("Generating initial master encryption key...")
            master_key_id = await key_management_service.generate_master_key(db)
            logger.info(f"Master key generated: {master_key_id}")
            
            # 3. Generate data encryption keys for different purposes
            logger.info("Generating data encryption keys...")
            
            purposes = [
                "user_data",
                "submissions",
                "audit_logs",
                "session_data",
                "blockchain_data"
            ]
            
            for purpose in purposes:
                key_id = await key_management_service.generate_data_encryption_key(db, purpose)
                logger.info(f"Data encryption key generated for {purpose}: {key_id}")
            
            # 4. Add some basic threat intelligence indicators
            logger.info("Adding basic threat intelligence indicators...")
            
            # Add some common malicious IP ranges (examples)
            threat_indicators = [
                {
                    "indicator_type": "IP",
                    "indicator_value": "127.0.0.1",
                    "threat_type": "test_indicator",
                    "severity": "LOW",
                    "source": "initialization",
                    "confidence": 50
                }
            ]
            
            for indicator in threat_indicators:
                await security_service.add_threat_intelligence(db, **indicator)
            
            # 5. Create initial data processing records for GDPR compliance
            logger.info("Creating initial data processing records...")
            
            # This would typically be done through the admin interface
            # but we can create basic records here
            
            logger.info("Security system initialization completed successfully!")
            
            # 6. Display security statistics
            logger.info("Security system statistics:")
            
            key_stats = await key_management_service.get_key_statistics(db)
            logger.info(f"  - Active encryption keys: {key_stats.get('active_keys', {})}")
            logger.info(f"  - Total keys: {key_stats.get('total_keys', 0)}")
            
            security_dashboard = await security_service.get_security_dashboard_data(db)
            logger.info(f"  - Active threat indicators: {security_dashboard.get('active_threat_indicators', 0)}")
            
            compliance_dashboard = await compliance_service.get_compliance_dashboard_data(db)
            logger.info(f"  - Active retention policies: {compliance_dashboard.get('active_policies_count', 0)}")
            
    except Exception as e:
        logger.error(f"Error during security system initialization: {e}")
        raise


async def verify_security_setup():
    """Verify that the security system is properly set up."""
    
    logger.info("Verifying security system setup...")
    
    try:
        async with get_db() as db:
            # Check encryption service
            encryption_service = get_encryption_service()
            test_data = "Test encryption data"
            encrypted = encryption_service.encrypt_text(test_data)
            decrypted = encryption_service.decrypt_text(encrypted)
            
            if decrypted == test_data:
                logger.info("✓ Encryption service working correctly")
            else:
                logger.error("✗ Encryption service test failed")
                return False
            
            # Check key management
            active_master_key = await key_management_service.get_active_key(db, "master")
            if active_master_key:
                logger.info(f"✓ Active master key found: {active_master_key}")
            else:
                logger.error("✗ No active master key found")
                return False
            
            # Check data encryption keys
            active_data_key = await key_management_service.get_active_key(db, "data")
            if active_data_key:
                logger.info(f"✓ Active data key found: {active_data_key}")
            else:
                logger.warning("⚠ No active data key found (this may be normal)")
            
            # Check compliance policies
            compliance_data = await compliance_service.get_compliance_dashboard_data(db)
            if compliance_data.get('active_policies_count', 0) > 0:
                logger.info(f"✓ Data retention policies active: {compliance_data['active_policies_count']}")
            else:
                logger.error("✗ No active data retention policies found")
                return False
            
            logger.info("Security system verification completed successfully!")
            return True
            
    except Exception as e:
        logger.error(f"Error during security system verification: {e}")
        return False


async def generate_security_report():
    """Generate a security configuration report."""
    
    logger.info("Generating security configuration report...")
    
    try:
        async with get_db() as db:
            # Get comprehensive security statistics
            key_stats = await key_management_service.get_key_statistics(db)
            security_dashboard = await security_service.get_security_dashboard_data(db)
            compliance_dashboard = await compliance_service.get_compliance_dashboard_data(db)
            
            report = f"""
=== STYLOS SECURITY CONFIGURATION REPORT ===

Encryption Keys:
  - Total Keys: {key_stats.get('total_keys', 0)}
  - Active Keys by Type: {key_stats.get('active_keys', {})}
  - Expiring Keys: {key_stats.get('expiring_keys_count', 0)}

Security Monitoring:
  - Recent Incidents: {len(security_dashboard.get('recent_incidents', []))}
  - Active Threat Indicators: {security_dashboard.get('active_threat_indicators', 0)}
  - Failed Logins (24h): {security_dashboard.get('failed_logins_24h', 0)}
  - Security Status: {security_dashboard.get('security_status', 'UNKNOWN')}

Compliance:
  - Pending Data Subject Requests: {len(compliance_dashboard.get('pending_requests', []))}
  - Active Retention Policies: {compliance_dashboard.get('active_policies_count', 0)}
  - Compliance Status: {compliance_dashboard.get('compliance_status', 'UNKNOWN')}

Configuration:
  - Encryption Enabled: {os.getenv('ENABLE_DATA_ENCRYPTION', 'true')}
  - Intrusion Detection: {os.getenv('ENABLE_INTRUSION_DETECTION', 'true')}
  - Security Monitoring: {os.getenv('SECURITY_MONITORING_ENABLED', 'true')}
  - GDPR Compliance: {os.getenv('GDPR_COMPLIANCE_ENABLED', 'true')}
  - FERPA Compliance: {os.getenv('FERPA_COMPLIANCE_ENABLED', 'true')}

=== END REPORT ===
            """
            
            print(report)
            
            # Save report to file
            report_file = backend_dir / "security_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            
            logger.info(f"Security report saved to: {report_file}")
            
    except Exception as e:
        logger.error(f"Error generating security report: {e}")


async def main():
    """Main function to run security initialization."""
    
    if len(sys.argv) < 2:
        print("Usage: python init_security.py [init|verify|report]")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "init":
        await initialize_security_system()
    elif command == "verify":
        success = await verify_security_setup()
        sys.exit(0 if success else 1)
    elif command == "report":
        await generate_security_report()
    else:
        print("Invalid command. Use: init, verify, or report")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())