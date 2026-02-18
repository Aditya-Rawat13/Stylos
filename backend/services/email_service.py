"""
Email service for authentication-related notifications.
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from jinja2 import Environment, FileSystemLoader
import os

from core.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Email service for sending authentication-related emails."""
    
    def __init__(self):
        self.smtp_server = getattr(settings, 'SMTP_SERVER', 'localhost')
        self.smtp_port = getattr(settings, 'SMTP_PORT', 587)
        self.smtp_username = getattr(settings, 'SMTP_USERNAME', '')
        self.smtp_password = getattr(settings, 'SMTP_PASSWORD', '')
        self.from_email = getattr(settings, 'FROM_EMAIL', 'noreply@stylos.edu')
        self.use_tls = getattr(settings, 'SMTP_USE_TLS', True)
        
        # Initialize Jinja2 environment for email templates
        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates', 'emails')
        if os.path.exists(template_dir):
            self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        else:
            self.jinja_env = None
            logger.warning(f"Email template directory not found: {template_dir}")
    
    async def send_email(
        self, 
        to_email: str, 
        subject: str, 
        html_content: str, 
        text_content: Optional[str] = None
    ) -> bool:
        """Send an email."""
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = to_email
            
            # Add text content if provided
            if text_content:
                text_part = MIMEText(text_content, 'plain')
                msg.attach(text_part)
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                
                server.send_message(msg)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False
    
    def _render_template(self, template_name: str, **kwargs) -> str:
        """Render email template."""
        if not self.jinja_env:
            return self._get_fallback_template(template_name, **kwargs)
        
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            return self._get_fallback_template(template_name, **kwargs)
    
    def _get_fallback_template(self, template_name: str, **kwargs) -> str:
        """Get fallback template when Jinja2 templates are not available."""
        if template_name == 'verification.html':
            return f"""
            <html>
                <body>
                    <h2>Verify Your Email Address</h2>
                    <p>Hello {kwargs.get('full_name', 'User')},</p>
                    <p>Thank you for registering with Stylos. Please click the link below to verify your email address:</p>
                    <p><a href="{kwargs.get('verification_url')}">Verify Email Address</a></p>
                    <p>This link will expire in 24 hours.</p>
                    <p>If you didn't create an account, please ignore this email.</p>
                    <p>Best regards,<br>The Stylos Team</p>
                </body>
            </html>
            """
        elif template_name == 'password_reset.html':
            return f"""
            <html>
                <body>
                    <h2>Password Reset Request</h2>
                    <p>Hello {kwargs.get('full_name', 'User')},</p>
                    <p>You requested a password reset for your Stylos account. Click the link below to reset your password:</p>
                    <p><a href="{kwargs.get('reset_url')}">Reset Password</a></p>
                    <p>This link will expire in 1 hour.</p>
                    <p>If you didn't request this reset, please ignore this email.</p>
                    <p>Best regards,<br>The Stylos Team</p>
                </body>
            </html>
            """
        elif template_name == 'welcome.html':
            return f"""
            <html>
                <body>
                    <h2>Welcome to Stylos!</h2>
                    <p>Hello {kwargs.get('full_name', 'User')},</p>
                    <p>Your email has been verified successfully. Welcome to Stylos - Academic Writing Verification System!</p>
                    <p>You can now:</p>
                    <ul>
                        <li>Upload your writing samples to build your unique writing profile</li>
                        <li>Submit essays for authorship verification</li>
                        <li>View your blockchain portfolio of verified work</li>
                    </ul>
                    <p><a href="{kwargs.get('dashboard_url', '#')}">Go to Dashboard</a></p>
                    <p>Best regards,<br>The Stylos Team</p>
                </body>
            </html>
            """
        else:
            return f"<html><body><p>Email notification from Stylos</p></body></html>"
    
    async def send_verification_email(
        self, 
        to_email: str, 
        full_name: str, 
        verification_token: str
    ) -> bool:
        """Send email verification email."""
        verification_url = f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/verify-email?token={verification_token}"
        
        html_content = self._render_template(
            'verification.html',
            full_name=full_name,
            verification_url=verification_url
        )
        
        return await self.send_email(
            to_email=to_email,
            subject="Verify Your Stylos Account",
            html_content=html_content
        )
    
    async def send_password_reset_email(
        self, 
        to_email: str, 
        full_name: str, 
        reset_token: str
    ) -> bool:
        """Send password reset email."""
        reset_url = f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/reset-password?token={reset_token}"
        
        html_content = self._render_template(
            'password_reset.html',
            full_name=full_name,
            reset_url=reset_url
        )
        
        return await self.send_email(
            to_email=to_email,
            subject="Reset Your Stylos Password",
            html_content=html_content
        )
    
    async def send_welcome_email(
        self, 
        to_email: str, 
        full_name: str
    ) -> bool:
        """Send welcome email after email verification."""
        dashboard_url = f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/dashboard"
        
        html_content = self._render_template(
            'welcome.html',
            full_name=full_name,
            dashboard_url=dashboard_url
        )
        
        return await self.send_email(
            to_email=to_email,
            subject="Welcome to Stylos!",
            html_content=html_content
        )
    
    async def send_security_alert(
        self, 
        to_email: str, 
        full_name: str, 
        alert_type: str, 
        details: str
    ) -> bool:
        """Send security alert email."""
        html_content = f"""
        <html>
            <body>
                <h2>Security Alert</h2>
                <p>Hello {full_name},</p>
                <p>We detected a security event on your Stylos account:</p>
                <p><strong>Alert Type:</strong> {alert_type}</p>
                <p><strong>Details:</strong> {details}</p>
                <p>If this was you, no action is needed. If you didn't perform this action, please contact support immediately.</p>
                <p>Best regards,<br>The Stylos Security Team</p>
            </body>
        </html>
        """
        
        return await self.send_email(
            to_email=to_email,
            subject=f"Stylos Security Alert: {alert_type}",
            html_content=html_content
        )


# Global email service instance
email_service = EmailService()