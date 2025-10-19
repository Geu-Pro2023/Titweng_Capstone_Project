#!/bin/bash
# Titweng Cattle System - Local Development Server
echo "🐄 Starting Titweng Cattle System..."
echo "📡 Server will be available at: http://127.0.0.1:8000"
echo "📚 API Documentation: http://127.0.0.1:8000/docs"
echo "👤 Admin Login: username=admin, password=admin123"
echo ""
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000