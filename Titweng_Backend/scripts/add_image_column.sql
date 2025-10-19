-- Add image_data column to embeddings table
-- Run this as database owner or superuser

ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS image_data BYTEA;

-- Verify column was added
\d embeddings;