#!/usr/bin/env python3
"""
Quick script to create multiple admins for Titweng Capstone Project
Run this after your backend is deployed
"""

import requests
import json

# Your Render backend URL
API_BASE = "https://titweng-capstone-project.onrender.com"

# Single admin for capstone demonstration
# Additional admins can be created through the dashboard as needed
print("ℹ️  For capstone project: Using single admin account")
print("ℹ️  Additional admins can be created through dashboard when needed")
print("ℹ️  Login credentials: titweng / titweng")
print("")
print("🚀 Capstone project ready with simplified admin setup!")

if __name__ == "__main__":
    print("🎓 Titweng Capstone Project - Single Admin Setup")
    print(f"📡 Backend URL: {API_BASE}")
    print()
    print("📋 Login Credentials:")
    print("Username: titweng")
    print("Password: titweng@2025")
    print()
    print("ℹ️  This capstone project uses a single admin account for demonstration.")
    print("ℹ️  In production, additional admins would be created based on organizational needs.")
    print("ℹ️  The dashboard includes functionality to create additional admins when required.")