#!/usr/bin/env python3
"""
Titweng Cattle Recognition System - Authentication Module

This module handles admin authentication for the dashboard using JWT tokens.
It provides secure login, token verification, and admin account management.

Author: Geu Augustine
Project: Capstone Project - Cattle Identification System
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
import hashlib
import os
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from app.database import Base, get_db

# JWT Configuration for secure authentication
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "titweng-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# HTTP Bearer token security scheme
security = HTTPBearer()

class Admin(Base):
    """
    Admin user model for dashboard authentication.
    
    This model stores administrator account information including
    credentials, roles, and login tracking.
    """
    __tablename__ = "admins"
    
    admin_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)  # SHA256 hashed password
    full_name = Column(String, nullable=False)
    role = Column(String, default="admin")  # admin, super_admin
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

# Pydantic models for API request/response validation
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str
    user: dict

class AdminCreate(BaseModel):
    """Admin creation request model."""
    username: str
    email: str
    password: str
    full_name: str
    role: str = "admin"

# Password security functions
def hash_password(plain_password: str) -> str:
    """
    Hash password using SHA256 for secure storage.
    
    Args:
        plain_password (str): Plain text password
        
    Returns:
        str: SHA256 hashed password
    """
    return hashlib.sha256(plain_password.encode()).hexdigest()

def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verify plain password against stored hash.
    
    Args:
        plain_password (str): Plain text password to verify
        stored_hash (str): Stored password hash
        
    Returns:
        bool: True if password matches, False otherwise
    """
    return hash_password(plain_password) == stored_hash

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_admin(username: str = Depends(verify_token), db: Session = Depends(get_db)):
    """Get current authenticated admin"""
    admin = db.query(Admin).filter(Admin.username == username).first()
    if admin is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin not found"
        )
    return admin

def create_default_admin(database_session: Session):
    """
    Create default admin account if no admins exist.
    
    This function creates a default administrator account for initial
    system setup. It only creates the account if no other admins exist.
    
    Args:
        database_session (Session): Database session for admin creation
    """
    try:
        # Check if any admin accounts exist
        existing_admin_count = database_session.query(Admin).count()
        
        if existing_admin_count == 0:
            # Create default admin account
            default_admin = Admin(
                username="titweng",
                email="admin@titweng.com",
                password_hash=hash_password("titweng@2025"),
                full_name="Titweng System Administrator",
                role="super_admin"
            )
            
            database_session.add(default_admin)
            database_session.commit()
            
            print("SUCCESS: Default admin account created")
            print("Login credentials: titweng / titweng@2025")
        else:
            print(f"INFO: {existing_admin_count} admin account(s) already exist")
            
    except Exception as admin_creation_error:
        print(f"ERROR: Failed to create default admin - {str(admin_creation_error)}")
        database_session.rollback()