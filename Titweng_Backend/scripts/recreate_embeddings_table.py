#!/usr/bin/env python3
"""Recreate embeddings table with image_data column"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from dotenv import load_dotenv

load_dotenv()

def recreate_embeddings_table():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    try:
        print("🗑️  Dropping embeddings table...")
        cursor.execute("DROP TABLE IF EXISTS embeddings CASCADE;")
        
        print("📋 Creating embeddings table with image_data column...")
        cursor.execute("""
            CREATE TABLE embeddings (
                embedding_id SERIAL PRIMARY KEY,
                cow_id INTEGER REFERENCES cows(cow_id) ON DELETE CASCADE,
                embedding VECTOR(256),
                image_path TEXT,
                image_data BYTEA,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("✅ Embeddings table recreated successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    recreate_embeddings_table()