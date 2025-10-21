#!/usr/bin/env python3
"""
Quick script to create multiple admins for Titweng Capstone Project
Run this after your backend is deployed
"""

import requests
import json

# Your Render backend URL
API_BASE = "https://titweng-capstone-project.onrender.com"

# Admin accounts to create
ADMINS = [
    {
        "username": "supervisor",
        "email": "supervisor@titweng.com",
        "password": "supervisor123",
        "full_name": "Project Supervisor",
        "role": "super_admin"
    },
    {
        "username": "operator1",
        "email": "operator1@titweng.com", 
        "password": "operator123",
        "full_name": "Field Operator 1",
        "role": "admin"
    },
    {
        "username": "operator2",
        "email": "operator2@titweng.com",
        "password": "operator123", 
        "full_name": "Field Operator 2",
        "role": "admin"
    }
]

def create_admin(admin_data):
    try:
        response = requests.post(f"{API_BASE}/auth/create-admin", json=admin_data)
        if response.status_code == 200:
            print(f"✅ Created admin: {admin_data['username']} ({admin_data['full_name']})")
        else:
            print(f"❌ Failed to create {admin_data['username']}: {response.text}")
    except Exception as e:
        print(f"❌ Error creating {admin_data['username']}: {e}")

if __name__ == "__main__":
    print("🔐 Creating additional admin accounts for Titweng Capstone Project...")
    print(f"📡 Backend URL: {API_BASE}")
    print()
    
    for admin in ADMINS:
        create_admin(admin)
    
    print()
    print("📋 Admin Login Credentials:")
    print("Default: admin / admin123")
    for admin in ADMINS:
        print(f"{admin['full_name']}: {admin['username']} / {admin['password']}")