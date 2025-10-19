-- Manual admin user creation for Render database
-- Run this in your Render database psql console

-- Create admin user with bcrypt hash of "admin123"
INSERT INTO users (username, email, password_hash, role) 
VALUES (
    'admin', 
    'admin@titweng.com', 
    '$2b$12$LQv3c1yqBwEHFuryHnS00O7B/JBGhlQGXTw7xz4Uy3Cd6AT2lSAyG', 
    'admin'
) ON CONFLICT (username) DO NOTHING;

-- Verify admin user was created
SELECT username, email, role FROM users WHERE role = 'admin';