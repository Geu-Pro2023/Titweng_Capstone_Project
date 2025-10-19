# Render PostgreSQL pgvector Extension Setup

## Problem
The application requires the `pgvector` extension for storing 256-dimensional embeddings, but Render PostgreSQL databases don't have this extension enabled by default.

## Solution Steps

### 1. Enable pgvector Extension on Render Database

**Option A: Via Render Dashboard (Recommended)**
1. Go to your Render Dashboard
2. Navigate to your PostgreSQL database service
3. Click on "Connect" to get connection details
4. Use a PostgreSQL client (like pgAdmin, DBeaver, or psql) to connect
5. Run this SQL command:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

**Option B: Via psql Command Line**
1. Get your database connection string from Render dashboard
2. Connect using psql:
   ```bash
   psql "your_render_database_connection_string"
   ```
3. Run the extension command:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   \q
   ```

### 2. Verify Extension Installation
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 3. Deploy Application
After enabling the extension, deploy your application. The startup sequence will:
1. Verify pgvector extension exists
2. Create all database tables with VECTOR(256) columns
3. Create admin user
4. Start the FastAPI server

## Alternative: Contact Render Support

If you cannot enable the extension yourself:
1. Contact Render support via their dashboard
2. Request pgvector extension to be enabled on your PostgreSQL database
3. Mention you need it for vector similarity search in your AI application

## Verification

Once deployed successfully, you should see these logs:
```
pgvector extension created/verified
Database tables created successfully
Admin user created successfully
Siamese model loaded
```

## Database Schema After Setup

The `embeddings` table will have:
- `embedding_id`: Primary key
- `cow_id`: Foreign key to cows table
- `embedding`: VECTOR(256) - stores 256-dimensional embeddings
- `image_path`: Text path to image file
- `image_data`: BYTEA - actual image binary data
- `created_at`: Timestamp