-- Initialize Render PostgreSQL database with pgvector extension
-- This script should be run once on the Render database

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension is installed
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Create tables will be handled by SQLAlchemy in the application