-- Migration: Add audit logs table for security monitoring
-- Created: 2024-01-01
-- Description: Creates audit_logs table for tracking security events and user activities

CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50) NOT NULL,
    description TEXT NOT NULL,
    ip_address VARCHAR(45),  -- IPv6 compatible
    user_agent TEXT,
    request_id VARCHAR(100),
    session_id VARCHAR(100),
    metadata JSONB,
    risk_level VARCHAR(20) DEFAULT 'LOW',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_logs_event_category ON audit_logs(event_category);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_risk_level ON audit_logs(risk_level);

-- Add comments
COMMENT ON TABLE audit_logs IS 'Security audit log for tracking user activities and system events';
COMMENT ON COLUMN audit_logs.event_type IS 'Specific type of event (LOGIN_SUCCESS, PASSWORD_CHANGE, etc.)';
COMMENT ON COLUMN audit_logs.event_category IS 'Category of event (AUTH, ACCESS, DATA, SYSTEM)';
COMMENT ON COLUMN audit_logs.risk_level IS 'Risk level of the event (LOW, MEDIUM, HIGH, CRITICAL)';
COMMENT ON COLUMN audit_logs.metadata IS 'Additional event-specific data in JSON format';