"""
Single Sign-On (SSO) Service

Provides SSO integration with LMS platforms using SAML, OAuth, and LTI protocols.
"""

import base64
import hashlib
import hmac
import json
import logging
import secrets
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple
from xml.etree import ElementTree as ET

import aiohttp
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from pydantic import BaseModel

from core.config import settings
from models.user import User
from schemas.lms import LTILaunchRequest, SSOConfiguration, LMSType

logger = logging.getLogger(__name__)


class SSOError(Exception):
    """Base exception for SSO errors"""
    pass


class SAMLError(SSOError):
    """SAML-specific errors"""
    pass


class OAuthError(SSOError):
    """OAuth-specific errors"""
    pass


class LTIError(SSOError):
    """LTI-specific errors"""
    pass


class SAMLResponse(BaseModel):
    """SAML response data"""
    user_id: str
    email: str
    name: str
    roles: list
    institution_id: str
    attributes: Dict[str, Any] = {}


class OAuthTokenResponse(BaseModel):
    """OAuth token response"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class LTIUser(BaseModel):
    """LTI user data"""
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    roles: list
    context_id: str
    context_title: Optional[str] = None
    resource_link_id: str
    tool_consumer_instance_guid: str


class SSOService:
    """Service for handling SSO integrations"""
    
    def __init__(self):
        self.saml_configs: Dict[str, SSOConfiguration] = {}
        self.oauth_configs: Dict[str, SSOConfiguration] = {}
        self.lti_configs: Dict[str, SSOConfiguration] = {}
    
    def register_saml_config(self, institution_id: str, config: SSOConfiguration):
        """Register SAML configuration for institution"""
        self.saml_configs[institution_id] = config
    
    def register_oauth_config(self, institution_id: str, config: SSOConfiguration):
        """Register OAuth configuration for institution"""
        self.oauth_configs[institution_id] = config
    
    def register_lti_config(self, institution_id: str, config: SSOConfiguration):
        """Register LTI configuration for institution"""
        self.lti_configs[institution_id] = config
    
    # SAML Implementation
    
    def generate_saml_request(self, institution_id: str, relay_state: Optional[str] = None) -> Tuple[str, str]:
        """Generate SAML authentication request"""
        config = self.saml_configs.get(institution_id)
        if not config:
            raise SAMLError(f"No SAML configuration found for institution: {institution_id}")
        
        # Generate request ID
        request_id = f"_{secrets.token_hex(16)}"
        
        # Create SAML AuthnRequest
        saml_request = f"""<?xml version="1.0" encoding="UTF-8"?>
<samlp:AuthnRequest xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                    ID="{request_id}"
                    Version="2.0"
                    IssueInstant="{datetime.utcnow().isoformat()}Z"
                    Destination="{config.saml_metadata_url}"
                    AssertionConsumerServiceURL="{settings.BASE_URL}/api/v1/lms/sso/saml/acs"
                    ProtocolBinding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST">
    <saml:Issuer>{config.saml_entity_id}</saml:Issuer>
    <samlp:NameIDPolicy Format="urn:oasis:names:tc:SAML:2.0:nameid-format:emailAddress" AllowCreate="true"/>
</samlp:AuthnRequest>"""
        
        # Base64 encode and URL encode
        encoded_request = base64.b64encode(saml_request.encode()).decode()
        
        # Build redirect URL
        params = {
            "SAMLRequest": encoded_request
        }
        if relay_state:
            params["RelayState"] = relay_state
        
        redirect_url = f"{config.saml_metadata_url}?{urllib.parse.urlencode(params)}"
        
        return redirect_url, request_id
    
    def validate_saml_response(self, saml_response: str, relay_state: Optional[str] = None) -> SAMLResponse:
        """Validate and parse SAML response"""
        try:
            # Decode base64
            decoded_response = base64.b64decode(saml_response)
            
            # Parse XML
            root = ET.fromstring(decoded_response)
            
            # Extract user information (simplified - in production, verify signatures)
            assertion = root.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Assertion")
            if assertion is None:
                raise SAMLError("No assertion found in SAML response")
            
            subject = assertion.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}Subject")
            name_id = subject.find(".//{urn:oasis:names:tc:SAML:2.0:assertion}NameID")
            
            if name_id is None:
                raise SAMLError("No NameID found in SAML response")
            
            user_id = name_id.text
            
            # Extract attributes
            attributes = {}
            attr_statements = assertion.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeStatement")
            for attr_statement in attr_statements:
                attrs = attr_statement.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}Attribute")
                for attr in attrs:
                    attr_name = attr.get("Name")
                    attr_values = [val.text for val in attr.findall(".//{urn:oasis:names:tc:SAML:2.0:assertion}AttributeValue")]
                    attributes[attr_name] = attr_values[0] if len(attr_values) == 1 else attr_values
            
            return SAMLResponse(
                user_id=user_id,
                email=attributes.get("email", user_id),
                name=attributes.get("displayName", ""),
                roles=attributes.get("roles", []),
                institution_id=attributes.get("institution", ""),
                attributes=attributes
            )
        
        except ET.ParseError as e:
            raise SAMLError(f"Invalid SAML XML: {str(e)}")
        except Exception as e:
            raise SAMLError(f"SAML validation error: {str(e)}")
    
    # OAuth Implementation
    
    def generate_oauth_url(self, institution_id: str, state: Optional[str] = None) -> str:
        """Generate OAuth authorization URL"""
        config = self.oauth_configs.get(institution_id)
        if not config:
            raise OAuthError(f"No OAuth configuration found for institution: {institution_id}")
        
        params = {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": f"{settings.BASE_URL}/api/v1/lms/sso/oauth/callback",
            "scope": "read:user read:courses",
            "state": state or secrets.token_urlsafe(32)
        }
        
        return f"{config.oauth_authorize_url}?{urllib.parse.urlencode(params)}"
    
    async def exchange_oauth_code(self, institution_id: str, code: str, state: str) -> OAuthTokenResponse:
        """Exchange OAuth authorization code for access token"""
        config = self.oauth_configs.get(institution_id)
        if not config:
            raise OAuthError(f"No OAuth configuration found for institution: {institution_id}")
        
        token_data = {
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "redirect_uri": f"{settings.BASE_URL}/api/v1/lms/sso/oauth/callback"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config.oauth_token_url, data=token_data) as response:
                if response.status == 200:
                    token_response = await response.json()
                    return OAuthTokenResponse(**token_response)
                else:
                    error_text = await response.text()
                    raise OAuthError(f"Token exchange failed: {error_text}")
    
    async def get_oauth_user_info(self, access_token: str, lms_type: LMSType, base_url: str) -> Dict[str, Any]:
        """Get user information using OAuth access token"""
        headers = {"Authorization": f"Bearer {access_token}"}
        
        # Different endpoints for different LMS types
        if lms_type == LMSType.CANVAS:
            user_url = f"{base_url}/api/v1/users/self"
        elif lms_type == LMSType.BLACKBOARD:
            user_url = f"{base_url}/learn/api/public/v1/users/me"
        else:
            raise OAuthError(f"OAuth user info not supported for LMS type: {lms_type}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(user_url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise OAuthError(f"Failed to get user info: {error_text}")
    
    # LTI Implementation
    
    def validate_lti_signature(self, launch_data: Dict[str, Any], signature: str, consumer_secret: str) -> bool:
        """Validate LTI OAuth 1.0 signature"""
        try:
            # Remove oauth_signature from parameters
            params = {k: v for k, v in launch_data.items() if k != "oauth_signature"}
            
            # Sort parameters
            sorted_params = sorted(params.items())
            
            # Create parameter string
            param_string = "&".join([f"{k}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_params])
            
            # Create signature base string
            base_string = f"POST&{urllib.parse.quote(settings.BASE_URL + '/api/v1/lms/lti/launch', safe='')}&{urllib.parse.quote(param_string, safe='')}"
            
            # Create signing key
            signing_key = f"{urllib.parse.quote(consumer_secret, safe='')}&"
            
            # Generate signature
            expected_signature = base64.b64encode(
                hmac.new(
                    signing_key.encode(),
                    base_string.encode(),
                    hashlib.sha1
                ).digest()
            ).decode()
            
            return hmac.compare_digest(signature, expected_signature)
        
        except Exception as e:
            logger.error(f"LTI signature validation error: {str(e)}")
            return False
    
    def parse_lti_launch(self, launch_data: Dict[str, Any]) -> LTIUser:
        """Parse LTI launch data"""
        try:
            # Extract required fields
            user_id = launch_data.get("user_id")
            if not user_id:
                raise LTIError("Missing required field: user_id")
            
            roles = launch_data.get("roles", "").split(",")
            context_id = launch_data.get("context_id", "")
            resource_link_id = launch_data.get("resource_link_id", "")
            tool_consumer_instance_guid = launch_data.get("tool_consumer_instance_guid", "")
            
            return LTIUser(
                user_id=user_id,
                email=launch_data.get("lis_person_contact_email_primary"),
                name=launch_data.get("lis_person_name_full"),
                roles=roles,
                context_id=context_id,
                context_title=launch_data.get("context_title"),
                resource_link_id=resource_link_id,
                tool_consumer_instance_guid=tool_consumer_instance_guid
            )
        
        except Exception as e:
            raise LTIError(f"Failed to parse LTI launch data: {str(e)}")
    
    def generate_lti_response(self, user: User, return_url: Optional[str] = None) -> str:
        """Generate LTI response HTML"""
        # Create JWT token for user
        token_data = {
            "sub": str(user.id),
            "email": user.email,
            "name": user.name,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        access_token = jwt.encode(token_data, settings.SECRET_KEY, algorithm="HS256")
        
        # Determine redirect URL
        redirect_url = return_url or f"{settings.FRONTEND_URL}/dashboard"
        redirect_url += f"?token={access_token}"
        
        # Generate HTML response
        html_response = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Project Stylos - LTI Launch</title>
    <script>
        window.location.href = "{redirect_url}";
    </script>
</head>
<body>
    <p>Redirecting to Project Stylos...</p>
    <p>If you are not redirected automatically, <a href="{redirect_url}">click here</a>.</p>
</body>
</html>
"""
        return html_response
    
    # Utility methods
    
    def create_sso_session(self, user_data: Dict[str, Any], institution_id: str) -> str:
        """Create SSO session token"""
        session_data = {
            "user_data": user_data,
            "institution_id": institution_id,
            "created_at": datetime.utcnow().isoformat(),
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        return jwt.encode(session_data, settings.SECRET_KEY, algorithm="HS256")
    
    def validate_sso_session(self, token: str) -> Dict[str, Any]:
        """Validate SSO session token"""
        try:
            return jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            raise SSOError("SSO session expired")
        except jwt.InvalidTokenError:
            raise SSOError("Invalid SSO session token")


# Global service instance
sso_service = SSOService()