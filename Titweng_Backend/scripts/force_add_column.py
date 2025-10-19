#!/usr/bin/env python3
"""Force add image_data column using SQLAlchemy"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from app.database import engine

def force_add_column():
    try:
        with engine.connect() as conn:
            # Check if column exists
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'embeddings' AND column_name = 'image_data'
            """))
            
            if result.fetchone():
                print("✅ image_data column already exists")
                return
            
            # Add column
            conn.execute(text("ALTER TABLE embeddings ADD COLUMN image_data BYTEA"))
            conn.commit()
            print("✅ Added image_data column successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("The embeddings table may need to be recreated by database admin")

if __name__ == "__main__":
    force_add_column()