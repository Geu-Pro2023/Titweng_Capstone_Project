#!/usr/bin/env python3
"""Database setup and admin creation script"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal, User, Base, engine
import bcrypt
from dotenv import load_dotenv

load_dotenv()

def setup_database():
    """Create all tables"""
    print("📋 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")

def create_admin():
    """Create admin user"""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if admin:
            print("✅ Admin user already exists")
            return
        
        admin_user = User(
            username=os.getenv("ADMIN_USERNAME", "admin"),
            email=os.getenv("ADMIN_EMAIL", "admin@example.com"),
            password_hash=bcrypt.hashpw(os.getenv("ADMIN_PASSWORD", "admin123").encode('utf-8'), bcrypt.gensalt()).decode('utf-8'),
            role="admin"
        )
        
        db.add(admin_user)
        db.commit()
        print("✅ Admin user created successfully!")
        print(f"   Username: {admin_user.username}")
        print(f"   Email: {admin_user.email}")
        
    except Exception as e:
        print(f"❌ Error creating admin: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    setup_database()
    create_admin()