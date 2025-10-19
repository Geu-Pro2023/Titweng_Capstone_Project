import os
from dotenv import load_dotenv
from app.database import SessionLocal, User
from passlib.context import CryptContext

# Load environment variables
load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def create_admin():
    db = SessionLocal()
    try:
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD", "admin123")
        email = os.getenv("ADMIN_EMAIL", "admin@titweng.com")

        # Check if admin exists
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            print(f"Admin user '{username}' already exists")
            return

        # Create admin user
        admin = User(
            username=username, 
            email=email, 
            password_hash=hash_password(password), 
            role="admin"
        )
        db.add(admin)
        db.commit()
        print(f"Admin user '{username}' created successfully")
        print(f"Email: {email}")
        print(f"Role: admin")
    finally:
        db.close()

def create_test_user():
    db = SessionLocal()
    try:
        # Check if test user exists
        existing = db.query(User).filter(User.username == "testuser").first()
        if existing:
            print("Test user already exists")
            return

        # Create test user
        user = User(
            username="testuser",
            email="user@titweng.com",
            password_hash=hash_password("user123"),
            role="user"
        )
        db.add(user)
        db.commit()
        print("Test user 'testuser' created successfully")
        print("Password: user123")
        print("Role: user")
    finally:
        db.close()

if __name__ == "__main__":
    print("Creating users...")
    create_admin()
    create_test_user()
    print("User creation complete!")
