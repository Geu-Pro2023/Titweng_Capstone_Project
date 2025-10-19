import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP, Float, Boolean, LargeBinary
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ================================
# DATABASE SETUP
# ================================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "titweng")
DB_USER = os.getenv("DB_USER", "titweng_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "titweng!.com")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ================================
# DATABASE MODELS
# ================================
class Owner(Base):
    __tablename__ = "owners"
    owner_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120))
    phone = Column(String(20))
    address = Column(Text)
    national_id = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())
    cows = relationship("Cow", back_populates="owner")

class Cow(Base):
    __tablename__ = "cows"
    cow_id = Column(Integer, primary_key=True, index=True)
    cow_tag = Column(String(50), unique=True, nullable=False)
    breed = Column(String(100))
    color = Column(String(50))
    age = Column(Integer)
    registration_date = Column(TIMESTAMP, server_default=func.now())
    owner_id = Column(Integer, ForeignKey("owners.owner_id", ondelete="CASCADE"))
    qr_code_link = Column(Text)
    receipt_pdf_link = Column(Text)
    owner = relationship("Owner", back_populates="cows")
    embeddings = relationship("Embedding", back_populates="cow")

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(120), unique=True)
    phone = Column(String(20))
    password_hash = Column(Text, nullable=False)
    role = Column(String(20), default="user")
    created_at = Column(TIMESTAMP, server_default=func.now())

class Embedding(Base):
    __tablename__ = "embeddings"
    embedding_id = Column(Integer, primary_key=True, index=True)
    cow_id = Column(Integer, ForeignKey("cows.cow_id", ondelete="CASCADE"))
    embedding = Column(Vector(256))
    image_path = Column(Text)  # URL or file path
    image_data = Column(LargeBinary)  # Store actual image bytes
    created_at = Column(TIMESTAMP, server_default=func.now())
    cow = relationship("Cow", back_populates="embeddings")

class Report(Base):
    __tablename__ = "reports"
    report_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    report_type = Column(String(50))
    description = Column(Text)
    location = Column(String(200))
    image_path = Column(Text)
    status = Column(String(20), default="pending")
    created_at = Column(TIMESTAMP, server_default=func.now())

class VerificationLog(Base):
    __tablename__ = "verification_logs"
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    cow_id = Column(Integer, ForeignKey("cows.cow_id"))
    verification_image = Column(Text)
    similarity_score = Column(Float)
    location = Column(Text)
    verified = Column(Boolean)
    created_at = Column(TIMESTAMP, server_default=func.now())

# ================================
# DATABASE FUNCTIONS
# ================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()