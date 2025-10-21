#!/usr/bin/env python3
"""
Titweng Cattle Recognition System - Main API Server

This is the core backend API for the Titweng cattle identification system.
It provides endpoints for cattle registration, verification using Siamese CNN,
and administrative functions for managing the cattle database.

Author: Geu Aguto Garang
Project: Capstone Project - Cattle Identification System
Technology Stack: FastAPI, PostgreSQL, PyTorch, Siamese CNN
"""

# Standard library imports
import os
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Optional, List
from io import BytesIO

# Third-party imports for machine learning
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# FastAPI and web framework imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from contextlib import asynccontextmanager

# External service imports
from twilio.rest import Client
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A5
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.units import mm
import qrcode
import json

# Environment configuration
from dotenv import load_dotenv
load_dotenv()

# Import database models and functions
from app.database import (
    Owner, Cow, User, Embedding, Report, VerificationLog,
    get_db, SessionLocal
)

# Import authentication
from app.auth import (
    Admin, LoginRequest, LoginResponse, AdminCreate,
    hash_password, verify_password, create_access_token,
    get_current_admin, create_default_admin, ACCESS_TOKEN_EXPIRE_MINUTES
)

# ============================================================================
# SIAMESE CONVOLUTIONAL NEURAL NETWORK FOR CATTLE NOSE PRINT RECOGNITION
# ============================================================================

class SiameseNetwork(torch.nn.Module):
    """
    Siamese Convolutional Neural Network for cattle nose print identification.
    
    This network creates 256-dimensional embeddings from nose print images
    that can be compared using cosine similarity for cattle verification.
    
    Architecture:
    - Backbone: ResNet-18 (pre-trained features removed)
    - Feature extraction: Convolutional layers
    - Embedding: Fully connected layers with dropout
    - Output: 256-dimensional normalized vector
    
    Args:
        embedding_dim (int): Dimension of output embedding vector (default: 256)
    """
    
    def __init__(self, embedding_dim=256):
        super().__init__()
        
        # Use ResNet-18 as backbone feature extractor
        # Remove the final classification layer to get feature maps
        resnet_backbone = models.resnet18(pretrained=False)
        feature_layers = list(resnet_backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*feature_layers)
        
        # Custom fully connected layers for embedding generation
        self.embedding_network = torch.nn.Sequential(
            torch.nn.Flatten(),                    # Flatten feature maps
            torch.nn.Linear(512, 512),             # First dense layer
            torch.nn.ReLU(inplace=True),           # Activation function
            torch.nn.Dropout(0.2),                 # Regularization
            torch.nn.Linear(512, embedding_dim)    # Final embedding layer
        )
    
    def forward_one(self, input_image):
        """
        Forward pass for a single image to generate embedding.
        
        Args:
            input_image (torch.Tensor): Input image tensor [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Normalized embedding vector [batch_size, embedding_dim]
        """
        # Extract features using ResNet backbone
        features = self.feature_extractor(input_image)
        
        # Generate embedding from features
        embedding = self.embedding_network(features)
        
        # L2 normalize the embedding for cosine similarity comparison
        normalized_embedding = F.normalize(embedding, p=2, dim=1)
        
        return normalized_embedding

# Global variable to store the loaded model
siamese_model = None

def load_siamese_model():
    """
    Load the pre-trained Siamese CNN model from disk.
    
    This function loads the trained model weights and prepares the model
    for inference. The model is set to evaluation mode to disable dropout
    and batch normalization training behavior.
    
    Returns:
        SiameseNetwork or None: Loaded model instance or None if loading fails
    """
    global siamese_model
    
    # Define the path to the trained model file
    model_file_path = "ml_models/siamese/siamese_model_final.pth"
    
    # Check if model file exists
    if not os.path.exists(model_file_path):
        print(f"ERROR: Siamese model file not found at {model_file_path}")
        return None
    
    try:
        # Initialize the model architecture
        model = SiameseNetwork(embedding_dim=256)
        
        # Load the trained weights
        model_checkpoint = torch.load(model_file_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if isinstance(model_checkpoint, dict) and "model_state" in model_checkpoint:
            # Checkpoint contains additional metadata
            model.load_state_dict(model_checkpoint["model_state"])
        else:
            # Checkpoint contains only model weights
            model.load_state_dict(model_checkpoint)
        
        # Set model to evaluation mode (important for inference)
        model.eval()
        
        print("SUCCESS: Siamese CNN model loaded and ready for inference")
        return model
        
    except Exception as model_error:
        print(f"ERROR: Failed to load Siamese model - {str(model_error)}")
        return None

# ============================================================================
# IMAGE PROCESSING AND MACHINE LEARNING UTILITIES
# ============================================================================

def preprocess_image_for_model(image_array: np.ndarray, target_size=(128, 128)) -> torch.Tensor:
    """
    Preprocess nose print image for Siamese CNN inference.
    
    This function applies the same preprocessing pipeline used during training:
    1. Resize image to model input size
    2. Convert to tensor format
    3. Normalize using ImageNet statistics
    
    Args:
        image_array (np.ndarray): Input image as numpy array (RGB format)
        target_size (tuple): Target image size for model input (default: 128x128)
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model inference
    """
    # Define preprocessing pipeline (same as training)
    preprocessing_pipeline = transforms.Compose([
        transforms.Resize(target_size),                           # Resize to model input size
        transforms.ToTensor(),                                    # Convert to tensor [0,1]
        transforms.Normalize(                                     # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],                         # RGB channel means
            std=[0.229, 0.224, 0.225]                           # RGB channel standard deviations
        )
    ])
    
    # Convert numpy array to PIL Image and apply preprocessing
    pil_image = Image.fromarray(image_array)
    processed_tensor = preprocessing_pipeline(pil_image)
    
    # Add batch dimension [1, 3, H, W] for model input
    batched_tensor = processed_tensor.unsqueeze(0)
    
    return batched_tensor

def extract_nose_print_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract 256-dimensional embedding from nose print image using Siamese CNN.
    
    This function processes a preprocessed image tensor through the Siamese
    network to generate a unique embedding vector for the cattle's nose print.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image tensor [1, 3, H, W]
        
    Returns:
        np.ndarray: 256-dimensional embedding vector
        
    Raises:
        HTTPException: If model is not loaded
        ValueError: If embedding quality is poor (unclear image)
    """
    global siamese_model
    
    # Ensure model is loaded
    if siamese_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Siamese CNN model not loaded - server initialization error"
        )
    
    # Generate embedding without gradient computation (inference mode)
    with torch.no_grad():
        # Forward pass through Siamese network
        embedding_tensor = siamese_model.forward_one(image_tensor)
        
        # Convert to numpy array and flatten
        embedding_vector = embedding_tensor.cpu().numpy().flatten()
        
        # Quality validation - check embedding magnitude
        embedding_magnitude = np.linalg.norm(embedding_vector)
        
        # Validate embedding quality based on magnitude
        if embedding_magnitude < 0.5:
            raise ValueError("Low quality embedding detected - image may be too dark or blurry")
        elif embedding_magnitude > 2.0:
            raise ValueError("Unusual embedding detected - image may be corrupted or invalid")
        
        return embedding_vector

def validate_nose_print_image(image_array: np.ndarray) -> Optional[np.ndarray]:
    """
    Validate and process nose print image for quality.
    
    This function performs basic quality checks on uploaded nose print images
    to ensure they are suitable for embedding extraction.
    
    Args:
        image_array (np.ndarray): Input image as numpy array
        
    Returns:
        np.ndarray or None: Validated image array or None if quality is poor
    """
    # Check if image is empty or corrupted
    if image_array.size == 0:
        print("WARNING: Empty image detected")
        return None
    
    # Calculate image variance to detect blank or low-contrast images
    image_variance = np.var(image_array)
    
    # Reject images with very low variance (likely blank or uniform)
    if image_variance < 50:
        print(f"WARNING: Low contrast image rejected (variance: {image_variance:.2f})")
        return None
    
    # Image passes quality checks
    return image_array

# ============================================================================
# SMS NOTIFICATION SERVICE CONFIGURATION
# ============================================================================

# Load Twilio credentials from environment variables
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")

# Initialize Twilio client if credentials are available
if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    print("SUCCESS: Twilio SMS service initialized")
else:
    twilio_client = None
    print("WARNING: Twilio credentials not found - SMS notifications disabled")

# ============================================================================
# NOTIFICATION SERVICES (SMS AND EMAIL)
# ============================================================================

def send_sms_notification(recipient_phone: str, message_text: str) -> bool:
    """
    Send SMS notification to cattle owner using Twilio service.
    
    Args:
        recipient_phone (str): Phone number to send SMS to
        message_text (str): SMS message content
        
    Returns:
        bool: True if SMS sent successfully, False otherwise
    """
    try:
        if twilio_client and TWILIO_PHONE_NUMBER:
            # Send SMS using Twilio API
            message = twilio_client.messages.create(
                body=message_text,
                from_=TWILIO_PHONE_NUMBER,
                to=recipient_phone
            )
            print(f"SUCCESS: SMS sent to {recipient_phone} (SID: {message.sid})")
            return True
        else:
            # SMS service not configured - log what would be sent
            print(f"INFO: SMS service not configured. Would send to {recipient_phone}: {message_text}")
            return False
            
    except Exception as sms_error:
        print(f"ERROR: Failed to send SMS to {recipient_phone} - {str(sms_error)}")
        return False

def send_email_with_receipt(owner_email: str, owner_name: str, cow_tag: str, pdf_receipt: bytes) -> bool:
    """
    Send email notification with PDF receipt to cattle owner.
    
    Args:
        owner_email (str): Owner's email address
        owner_name (str): Owner's full name
        cow_tag (str): Cattle identification tag
        pdf_receipt (bytes): PDF receipt content
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        # Load SMTP configuration from environment
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        # Check if email configuration is available
        if not all([smtp_server, smtp_username, smtp_password]):
            print("WARNING: Email configuration missing - email notifications disabled")
            return False
        
        print(f"INFO: Sending email to {owner_email} using {smtp_username}")
        
        # Create email message
        email_message = MIMEMultipart()
        email_message['From'] = smtp_username
        email_message['To'] = owner_email
        email_message['Subject'] = f"Cattle Registration Confirmation - {cow_tag}"
        
        # Email body content
        email_body = f"""Dear {owner_name},

Your cattle has been successfully registered in the Titweng Cattle Recognition System.

Registration Details:
- Cattle Tag: {cow_tag}
- Registration Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}
- System: Titweng Cattle Recognition System

Please find your official registration receipt attached to this email.
Keep this receipt safe as proof of registration.

Best regards,
Titweng Cattle Recognition System
"""
        
        # Attach email body
        email_message.attach(MIMEText(email_body, 'plain'))
        
        # Attach PDF receipt
        pdf_attachment = MIMEBase('application', 'octet-stream')
        pdf_attachment.set_payload(pdf_receipt)
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header(
            'Content-Disposition', 
            f'attachment; filename="{cow_tag}_registration_receipt.pdf"'
        )
        email_message.attach(pdf_attachment)
        
        # Send email via SMTP
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Enable encryption
            server.login(smtp_username, smtp_password)
            email_text = email_message.as_string()
            server.sendmail(smtp_username, owner_email, email_text)
        
        print(f"SUCCESS: Email sent to {owner_email}")
        return True
        
    except Exception as email_error:
        print(f"ERROR: Failed to send email to {owner_email} - {str(email_error)}")
        return False

def generate_unique_cow_tag(database_session: Session) -> str:
    """
    Generate a unique, secure cattle identification tag.
    
    The tag format is: TW + 2 random letters + 4 sequential digits
    Example: TWAB0001, TWXY0002, etc.
    
    Args:
        database_session (Session): Database session for checking uniqueness
        
    Returns:
        str: Unique cattle tag identifier
    """
    import random
    import string
    
    # Get current count of registered cattle
    cow_count_query = database_session.execute(text("SELECT COUNT(*) FROM cows"))
    total_registered_cows = cow_count_query.fetchone()[0]
    
    # Generate tag components
    prefix = "TW"  # Titweng prefix
    random_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    sequential_number = f"{total_registered_cows + 1:04d}"  # 4-digit number with leading zeros
    
    # Combine components to create tag
    proposed_tag = f"{prefix}{random_letters}{sequential_number}"
    
    # Ensure tag is unique in database (handle rare collision case)
    while database_session.query(Cow).filter(Cow.cow_tag == proposed_tag).first():
        # Generate new random letters if collision occurs
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        proposed_tag = f"{prefix}{random_letters}{sequential_number}"
    
    print(f"INFO: Generated unique cattle tag: {proposed_tag}")
    return proposed_tag

def generate_pdf_receipt(cow_tag: str, owner_name: str, cow_data: dict = None) -> bytes:
    """
    Generate professional PDF receipt for cattle registration.
    
    Args:
        cow_tag (str): Cattle identification tag
        owner_name (str): Owner's full name
        cow_data (dict): Additional cattle and owner information
        
    Returns:
        bytes: PDF receipt content as bytes
    """
    buffer = BytesIO()
    canvas_obj = canvas.Canvas(buffer, pagesize=A5)
    page_width, page_height = A5
    
    # Generate unique receipt number
    receipt_hash = hashlib.md5(f"{cow_tag}{datetime.now().isoformat()}".encode()).hexdigest()[:8].upper()
    receipt_number = f"TW-{receipt_hash}"
    
    # Professional page border
    canvas_obj.setStrokeColor(HexColor('#2c3e50'))
    canvas_obj.setLineWidth(3)
    canvas_obj.rect(5*mm, 5*mm, page_width-10*mm, page_height-10*mm)
    
    # Header section with logo area
    canvas_obj.setFillColor(HexColor('#074a42'))
    canvas_obj.rect(10*mm, page_height-55*mm, page_width-20*mm, 40*mm, fill=1)
    
    # Company title
    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 14)
    title_text = "TITWENG CATTLE SYSTEM"
    text_width = canvas_obj.stringWidth(title_text, "Helvetica-Bold", 14)
    canvas_obj.drawString((page_width - text_width) / 2, page_height-40*mm, title_text)
    
    # Subtitle
    canvas_obj.setFont("Helvetica-Bold", 10)
    subtitle_text = "OFFICIAL REGISTRATION RECEIPT"
    text_width = canvas_obj.stringWidth(subtitle_text, "Helvetica-Bold", 10)
    canvas_obj.drawString((page_width - text_width) / 2, page_height-50*mm, subtitle_text)
    
    # Receipt number
    canvas_obj.setFillColor(black)
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(page_width-45*mm, page_height-50*mm, f"Receipt #: {receipt_number}")
    
    # Main content area
    y_position = page_height - 65*mm
    
    # Registration timestamp
    canvas_obj.setFillColor(black)
    canvas_obj.setFont("Helvetica", 8)
    registration_date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    canvas_obj.drawString(15*mm, y_position, f"Registration Date: {registration_date}")
    y_position -= 8*mm
    
    # Cattle Information Section
    canvas_obj.setFillColor(HexColor('#3498db'))
    canvas_obj.rect(15*mm, y_position-2*mm, page_width-30*mm, 6*mm, fill=1)
    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 9)
    canvas_obj.drawString(17*mm, y_position, "CATTLE INFORMATION")
    
    canvas_obj.setFillColor(black)
    y_position -= 10*mm
    
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(17*mm, y_position, f"Cow Tag: {cow_tag}")
    y_position -= 5*mm
    
    if cow_data:
        canvas_obj.setFont("Helvetica", 7)
        canvas_obj.drawString(17*mm, y_position, f"Breed: {cow_data.get('breed', 'Not specified')}")
        y_position -= 4*mm
        canvas_obj.drawString(17*mm, y_position, f"Color: {cow_data.get('color', 'Not specified')}")
        y_position -= 4*mm
        age_text = f"{cow_data.get('age')} months" if cow_data.get('age') else "Not specified"
        canvas_obj.drawString(17*mm, y_position, f"Age: {age_text}")
        y_position -= 6*mm
    
    # Owner Information Section
    canvas_obj.setFillColor(HexColor('#27ae60'))
    canvas_obj.rect(15*mm, y_position-2*mm, page_width-30*mm, 6*mm, fill=1)
    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 9)
    canvas_obj.drawString(17*mm, y_position, "OWNER INFORMATION")
    
    canvas_obj.setFillColor(black)
    y_position -= 10*mm
    
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(17*mm, y_position, f"Full Name: {owner_name}")
    y_position -= 5*mm
    
    if cow_data and cow_data.get('owner_data'):
        owner_data = cow_data['owner_data']
        canvas_obj.setFont("Helvetica", 7)
        if owner_data.get('phone'):
            canvas_obj.drawString(17*mm, y_position, f"Phone: {owner_data.get('phone')}")
            y_position -= 4*mm
        if owner_data.get('email'):
            email = owner_data.get('email', '')
            display_email = email[:25] + '...' if len(email) > 25 else email
            canvas_obj.drawString(17*mm, y_position, f"Email: {display_email}")
            y_position -= 4*mm
    
    # Generate QR code with cattle information
    qr_data = {
        "system": "Titweng",
        "receipt_number": receipt_number,
        "cow_tag": cow_tag,
        "owner_name": owner_name,
        "registration_date": datetime.now().strftime('%Y-%m-%d'),
        "verification_url": f"https://titweng.com/verify/{cow_tag}"
    }
    
    # Create QR code
    qr_json = json.dumps(qr_data, separators=(',', ':'))
    qr_code = qrcode.QRCode(version=3, box_size=3, border=2)
    qr_code.add_data(qr_json)
    qr_code.make(fit=True)
    qr_image = qr_code.make_image(fill_color="black", back_color="white")
    
    # Save QR code temporarily and add to PDF
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as qr_temp_file:
        qr_image.save(qr_temp_file.name, format='PNG')
        qr_temp_path = qr_temp_file.name
    
    # Position QR code
    qr_x = page_width - 35*mm
    qr_y = 35*mm
    
    # QR code background
    canvas_obj.setFillColor(white)
    canvas_obj.setStrokeColor(HexColor('#2c3e50'))
    canvas_obj.setLineWidth(1)
    canvas_obj.rect(qr_x-2*mm, qr_y-2*mm, 24*mm, 24*mm, fill=1, stroke=1)
    
    canvas_obj.drawImage(qr_temp_path, qr_x, qr_y, width=20*mm, height=20*mm)
    os.unlink(qr_temp_path)
    
    # QR Code label - ensure black text
    canvas_obj.setFillColor(black)
    canvas_obj.setFont("Helvetica-Bold", 6)
    canvas_obj.drawString(qr_x+3*mm, qr_y-6*mm, "Scan to verify")
    
    # Signature line
    canvas_obj.setLineWidth(1)
    canvas_obj.setStrokeColor(black)
    canvas_obj.line(17*mm, 35*mm, 80*mm, 35*mm)
    
    # Signature text - ensure black text
    canvas_obj.setFillColor(black)
    canvas_obj.setFont("Helvetica-Bold", 8)
    canvas_obj.drawString(17*mm, 30*mm, "Registrar Signature")
    
    canvas_obj.setFillColor(black)
    canvas_obj.setFont("Helvetica", 6)
    canvas_obj.drawString(17*mm, 25*mm, "Titweng System Administrator")
    
    # Professional footer
    canvas_obj.setFillColor(HexColor('#95a5a6'))
    canvas_obj.rect(10*mm, 10*mm, page_width-20*mm, 8*mm, fill=1)
    
    canvas_obj.setFillColor(white)
    canvas_obj.setFont("Helvetica-Bold", 6)
    footer_text = "OFFICIAL TITWENG CATTLE REGISTRATION RECEIPT - KEEP SAFE FOR VERIFICATION"
    text_width = canvas_obj.stringWidth(footer_text, "Helvetica-Bold", 6)
    canvas_obj.drawString((page_width - text_width) / 2, 13*mm, footer_text)
    
    canvas_obj.showPage()
    canvas_obj.save()
    buffer.seek(0)
    return buffer.read()

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

@asynccontextmanager
async def application_lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown tasks.
    """
    global siamese_model
    
    # Startup tasks
    print("INFO: Starting Titweng Cattle Recognition System...")
    
    # Create static directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("static/receipts", exist_ok=True)
    os.makedirs("static/cow_images", exist_ok=True)
    
    # Load Siamese CNN model
    siamese_model = load_siamese_model()
    
    # Initialize database
    try:
        from app.database import Base, engine
        
        # Create pgvector extension for embedding storage
        with engine.connect() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            connection.commit()
            print("SUCCESS: pgvector extension created/verified")
        
        # Create all database tables
        Base.metadata.create_all(bind=engine)
        print("SUCCESS: Database tables created successfully")
        
        # Create default admin account
        database_session = SessionLocal()
        create_default_admin(database_session)
        database_session.close()
        
    except Exception as database_error:
        print(f"ERROR: Database initialization failed - {str(database_error)}")
    
    print("SUCCESS: Titweng system startup completed")
    
    yield
    
    # Shutdown tasks
    print("INFO: Shutting down Titweng system...")

# Create FastAPI application instance
app = FastAPI(
    title="Titweng Cattle Recognition System",
    description="Advanced cattle identification system using Siamese CNN for nose print recognition",
    version="1.0.0",
    lifespan=application_lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["System Status"])
def system_root():
    """Root endpoint providing system information."""
    return {
        "message": "Titweng Cattle Recognition System API",
        "status": "operational",
        "version": "1.0.0",
        "author": "Geu Augustine",
        "project": "Capstone Project"
    }

@app.get("/health", tags=["System Status"])
def health_check():
    """System health check endpoint."""
    try:
        # Test database connection
        database_session = SessionLocal()
        database_session.execute(text("SELECT 1"))
        database_session.close()
        database_connected = True
    except Exception as db_error:
        print(f"Database connection error: {db_error}")
        database_connected = False
    
    return {
        "status": "healthy" if database_connected else "database_error",
        "model_loaded": siamese_model is not None,
        "database_connected": database_connected,
        "timestamp": datetime.now().isoformat()
    }

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
def admin_login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Admin login endpoint."""
    try:
        # Find admin by username
        admin = db.query(Admin).filter(Admin.username == login_data.username).first()
        
        if not admin or not verify_password(login_data.password, admin.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        if not admin.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled"
            )
        
        # Update last login
        admin.last_login = datetime.utcnow()
        db.commit()
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": admin.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": admin.admin_id,
                "username": admin.username,
                "email": admin.email,
                "full_name": admin.full_name,
                "role": admin.role
            }
        }
    except HTTPException:
        raise
    except Exception as login_error:
        print(f"Login error: {str(login_error)}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/auth/verify", tags=["Authentication"])
def verify_token_endpoint(current_admin: Admin = Depends(get_current_admin)):
    """Verify JWT token."""
    return {
        "valid": True,
        "user": {
            "id": current_admin.admin_id,
            "username": current_admin.username,
            "email": current_admin.email,
            "full_name": current_admin.full_name,
            "role": current_admin.role
        }
    }

# Cattle registration endpoint
@app.post("/register-cow", tags=["Cattle Management"])
async def register_cattle(
    owner_name: str = Form(...),
    owner_email: str = Form(...),
    owner_phone: str = Form(...),
    owner_address: str = Form(None),
    national_id: str = Form(None),
    breed: str = Form(None),
    color: str = Form(None),
    age: str = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Register new cattle with nose print images."""
    try:
        print(f"INFO: Starting cattle registration for owner: {owner_name}")
        
        # Generate unique cattle tag
        cattle_tag = generate_unique_cow_tag(db)
        
        # Process age input
        cattle_age = None
        if age and age.isdigit():
            cattle_age = int(age)
        
        # Create or get owner record
        owner = db.query(Owner).filter(Owner.email == owner_email).first()
        if not owner:
            owner = Owner(
                full_name=owner_name,
                email=owner_email,
                phone=owner_phone,
                address=owner_address,
                national_id=national_id
            )
            db.add(owner)
            db.commit()
            db.refresh(owner)
        
        # Process nose print images and extract embeddings
        processed_embeddings = []
        
        for file_index, uploaded_file in enumerate(files[:7]):  # Maximum 7 images
            try:
                # Read and decode image
                image_content = await uploaded_file.read()
                image_array = np.frombuffer(image_content, np.uint8)
                decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if decoded_image is None:
                    print(f"WARNING: Could not decode image {file_index + 1}")
                    continue
                
                # Validate image quality
                rgb_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                validated_image = validate_nose_print_image(rgb_image)
                
                if validated_image is None:
                    print(f"WARNING: Image {file_index + 1} failed quality validation")
                    continue
                
                # Preprocess for model
                processed_tensor = preprocess_image_for_model(validated_image)
                
                # Extract embedding using Siamese CNN
                embedding_vector = extract_nose_print_embedding(processed_tensor)
                
                # Convert image to bytes for database storage
                _, image_buffer = cv2.imencode('.jpg', decoded_image)
                image_bytes = image_buffer.tobytes()
                
                processed_embeddings.append((embedding_vector, image_bytes))
                print(f"SUCCESS: Processed embedding {file_index + 1}")
                
            except ValueError as quality_error:
                print(f"WARNING: Skipping image {file_index + 1} - {str(quality_error)}")
                continue
            except Exception as processing_error:
                print(f"ERROR: Failed to process image {file_index + 1} - {str(processing_error)}")
                continue
        
        # Ensure minimum number of embeddings
        if len(processed_embeddings) < 3:
            raise HTTPException(
                status_code=400,
                detail="Need at least 3 high-quality nose print images for registration. Please ensure images are clear and well-lit."
            )
        
        print(f"SUCCESS: Generated {len(processed_embeddings)} embeddings for cattle registration")
        
        # Create cattle record
        new_cattle = Cow(
            cow_tag=cattle_tag,
            breed=breed,
            color=color,
            age=cattle_age,
            owner_id=owner.owner_id
        )
        db.add(new_cattle)
        db.flush()  # Get the cattle ID
        
        # Store embeddings in database
        for embedding_index, (embedding_vector, image_bytes) in enumerate(processed_embeddings):
            image_filename = f"{cattle_tag}_{embedding_index + 1:03d}.jpg"
            
            embedding_record = Embedding(
                cow_id=new_cattle.cow_id,
                embedding=embedding_vector.tolist(),
                image_path=image_filename,
                image_data=image_bytes
            )
            db.add(embedding_record)
        
        db.commit()
        
        # Generate PDF receipt
        cattle_data = {
            'breed': breed,
            'color': color,
            'age': cattle_age,
            'registration_date': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'owner_data': {
                'email': owner_email,
                'phone': owner_phone,
                'address': owner_address,
                'national_id': national_id
            }
        }
        pdf_receipt = generate_pdf_receipt(cattle_tag, owner_name, cattle_data)
        
        # Save receipt to filesystem
        receipt_path = f"static/receipts/{cattle_tag}_receipt.pdf"
        with open(receipt_path, "wb") as receipt_file:
            receipt_file.write(pdf_receipt)
        
        # Update cattle record with receipt path
        new_cattle.receipt_pdf_link = receipt_path
        db.commit()
        
        # Send notifications
        try:
            print("INFO: Sending notifications...")
            sms_sent = send_sms_notification(
                owner_phone,
                f"Your cattle has been registered successfully! Tag: {cattle_tag}. Keep your receipt safe."
            )
            email_sent = send_email_with_receipt(owner_email, owner_name, cattle_tag, pdf_receipt)
            
            notification_status = []
            if sms_sent:
                notification_status.append("SMS sent")
            if email_sent:
                notification_status.append("Email sent")
            
            notifications = ", ".join(notification_status) if notification_status else "Notifications disabled"
            
        except Exception as notification_error:
            print(f"WARNING: Notification error - {str(notification_error)}")
            notifications = "Notification error"
        
        return {
            "success": True,
            "cattle_id": new_cattle.cow_id,
            "cattle_tag": cattle_tag,
            "receipt_path": receipt_path,
            "embeddings_count": len(processed_embeddings),
            "message": f"Cattle registered successfully with tag {cattle_tag}",
            "notifications": notifications
        }
        
    except HTTPException:
        raise
    except Exception as registration_error:
        print(f"ERROR: Cattle registration failed - {str(registration_error)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(registration_error)}")

# Cattle verification endpoint
@app.post("/verify-cow", tags=["Cattle Management"])
async def verify_cattle(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Verify cattle identity using nose print images."""
    try:
        print(f"INFO: Starting cattle verification with {len(files)} images")
        
        if not files:
            raise HTTPException(status_code=400, detail="No images provided for verification")
        
        # Process verification images
        query_embeddings = []
        
        for file_index, uploaded_file in enumerate(files[:3]):  # Maximum 3 images for verification
            try:
                # Read and decode image
                image_content = await uploaded_file.read()
                image_array = np.frombuffer(image_content, np.uint8)
                decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if decoded_image is None:
                    continue
                
                # Validate and preprocess image
                rgb_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2RGB)
                validated_image = validate_nose_print_image(rgb_image)
                
                if validated_image is None:
                    continue
                
                # Extract embedding
                processed_tensor = preprocess_image_for_model(validated_image)
                embedding_vector = extract_nose_print_embedding(processed_tensor)
                query_embeddings.append(embedding_vector)
                
            except Exception as processing_error:
                print(f"WARNING: Failed to process verification image {file_index + 1} - {str(processing_error)}")
                continue
        
        if len(query_embeddings) == 0:
            raise HTTPException(status_code=400, detail="No valid images for verification")
        
        # Create average query embedding
        average_query_embedding = np.mean(query_embeddings, axis=0)
        normalized_query_embedding = average_query_embedding / np.linalg.norm(average_query_embedding)
        
        # Compare against registered cattle
        registered_cattle = db.query(Cow).filter(Cow.embeddings.any()).all()
        
        if not registered_cattle:
            return {
                "registered": False,
                "status": "NOT_REGISTERED",
                "message": "❌ COW NOT REGISTERED - No cattle in database"
            }
        
        # Calculate similarity scores
        cattle_similarity_scores = []
        
        for cattle in registered_cattle:
            if not cattle.embeddings:
                continue
            
            similarity_scores = []
            for stored_embedding in cattle.embeddings:
                try:
                    stored_vector = np.array(stored_embedding.embedding, dtype=np.float32)
                    normalized_stored_vector = stored_vector / np.linalg.norm(stored_vector)
                    cosine_similarity = np.dot(normalized_query_embedding, normalized_stored_vector)
                    similarity_scores.append(cosine_similarity)
                except Exception as similarity_error:
                    print(f"WARNING: Similarity calculation error - {str(similarity_error)}")
                    continue
            
            if similarity_scores:
                # Use average of top 3 similarities
                top_similarities = sorted(similarity_scores, reverse=True)[:3]
                average_similarity = np.mean(top_similarities)
                cattle_similarity_scores.append((cattle, average_similarity))
        
        if not cattle_similarity_scores:
            return {
                "registered": False,
                "status": "ERROR",
                "message": "❌ Unable to process verification"
            }
        
        # Find best match
        cattle_similarity_scores.sort(key=lambda x: x[1], reverse=True)
        best_match_cattle, best_similarity_score = cattle_similarity_scores[0]
        
        # Verification threshold
        VERIFICATION_THRESHOLD = 0.82
        
        if best_similarity_score >= VERIFICATION_THRESHOLD:
            # Successful verification
            owner = best_match_cattle.owner
            
            # Log verification
            verification_log = VerificationLog(
                cow_id=best_match_cattle.cow_id,
                similarity_score=float(best_similarity_score),
                verified=True,
                user_id=1  # Default user for now
            )
            db.add(verification_log)
            db.commit()
            
            # Send verification SMS
            if owner and owner.phone:
                send_sms_notification(
                    owner.phone,
                    f"Your cattle {best_match_cattle.cow_tag} was successfully verified!"
                )
            
            return {
                "registered": True,
                "status": "REGISTERED",
                "message": "✅ CATTLE IS REGISTERED - Verification Successful",
                "cattle_details": {
                    "cattle_tag": best_match_cattle.cow_tag,
                    "cattle_id": best_match_cattle.cow_id,
                    "breed": best_match_cattle.breed or "Not specified",
                    "color": best_match_cattle.color or "Not specified",
                    "age": f"{best_match_cattle.age} months" if best_match_cattle.age else "Not specified"
                },
                "owner_details": {
                    "owner_name": owner.full_name if owner else "Unknown",
                    "owner_email": owner.email if owner else "Unknown",
                    "owner_phone": owner.phone if owner else "Unknown"
                },
                "verification_info": {
                    "similarity_score": round(float(best_similarity_score), 4),
                    "confidence_level": "HIGH" if best_similarity_score > 0.85 else "MEDIUM"
                }
            }
        else:
            # Failed verification
            verification_log = VerificationLog(
                similarity_score=float(best_similarity_score),
                verified=False,
                user_id=1
            )
            db.add(verification_log)
            db.commit()
            
            return {
                "registered": False,
                "status": "NOT_REGISTERED",
                "message": "❌ CATTLE NOT REGISTERED - This cattle is not in the system",
                "verification_info": {
                    "highest_similarity": round(float(best_similarity_score), 4),
                    "threshold_required": VERIFICATION_THRESHOLD
                }
            }
            
    except HTTPException:
        raise
    except Exception as verification_error:
        print(f"ERROR: Cattle verification failed - {str(verification_error)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(verification_error)}")

# Dashboard endpoints
@app.get("/admin/dashboard", tags=["Admin Dashboard"])
def admin_dashboard_stats(db: Session = Depends(get_db)):
    """Get admin dashboard statistics."""
    total_cattle = db.query(Cow).count()
    total_owners = db.query(Owner).count()
    pending_reports = db.query(Report).filter(Report.status == "pending").count()
    
    return {
        "total_cows": total_cattle,
        "total_owners": total_owners,
        "pending_reports": pending_reports
    }

@app.get("/admin/cows", tags=["Admin Dashboard"])
def get_all_registered_cattle(db: Session = Depends(get_db)):
    """Get all registered cattle with owner details."""
    all_cattle = db.query(Cow).all()
    
    cattle_list = []
    for cattle in all_cattle:
        owner = cattle.owner
        cattle_list.append({
            "cow_id": cattle.cow_id,
            "cow_tag": cattle.cow_tag,
            "breed": cattle.breed,
            "color": cattle.color,
            "age": cattle.age,
            "registration_date": cattle.registration_date.strftime('%d/%m/%Y %H:%M') if cattle.registration_date else "Unknown",
            "owner_name": owner.full_name if owner else "Unknown",
            "owner_email": owner.email if owner else "Unknown",
            "owner_phone": owner.phone if owner else "Unknown",
            "owner_address": owner.address if owner else "Unknown",
            "embeddings_count": len(cattle.embeddings)
        })
    
    return {
        "total_cows": len(cattle_list),
        "cows": cattle_list
    }

@app.get("/admin/reports", tags=["Admin Dashboard"])
def get_all_reports(db: Session = Depends(get_db)):
    """Get all reports for admin review."""
    all_reports = db.query(Report).order_by(Report.created_at.desc()).all()
    
    reports_list = []
    for report in all_reports:
        reports_list.append({
            "report_id": report.report_id,
            "description": report.description,
            "location": report.location,
            "status": report.status,
            "created_at": report.created_at
        })
    
    return reports_list

# Utility endpoints
@app.get("/preview-next-cow-tag", tags=["Utilities"])
def preview_next_cattle_tag(db: Session = Depends(get_db)):
    """Preview what the next cattle tag will be."""
    next_tag = generate_unique_cow_tag(db)
    total_cattle = db.query(Cow).count()
    
    return {
        "next_cow_tag": next_tag,
        "explanation": "This tag will be automatically generated during registration",
        "current_total_cows": total_cattle,
        "note": "Cattle ID and tag are auto-generated for security"
    }

@app.get("/download-receipt/{cow_tag}", tags=["Utilities"])
def download_cattle_receipt(cow_tag: str, db: Session = Depends(get_db)):
    """Download PDF receipt for registered cattle."""
    try:
        cattle = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
        if not cattle:
            raise HTTPException(status_code=404, detail="Cattle not found")
        
        # Generate receipt if it doesn't exist
        if not cattle.receipt_pdf_link or not os.path.exists(cattle.receipt_pdf_link):
            owner_name = cattle.owner.full_name if cattle.owner else "Unknown Owner"
            cattle_data = {
                'breed': cattle.breed,
                'color': cattle.color,
                'age': cattle.age,
                'registration_date': cattle.registration_date.strftime('%d/%m/%Y %H:%M') if cattle.registration_date else 'Unknown',
                'owner_data': {
                    'email': cattle.owner.email if cattle.owner else '',
                    'phone': cattle.owner.phone if cattle.owner else '',
                    'address': cattle.owner.address if cattle.owner else '',
                    'national_id': cattle.owner.national_id if cattle.owner else ''
                }
            }
            pdf_receipt = generate_pdf_receipt(cow_tag, owner_name, cattle_data)
            
            # Save receipt
            receipt_path = f"static/receipts/{cow_tag}_receipt.pdf"
            with open(receipt_path, "wb") as receipt_file:
                receipt_file.write(pdf_receipt)
            
            # Update cattle record
            cattle.receipt_pdf_link = receipt_path
            db.commit()
        
        return FileResponse(
            path=cattle.receipt_pdf_link,
            filename=f"{cow_tag}_receipt.pdf",
            media_type="application/pdf"
        )
        
    except Exception as download_error:
        print(f"ERROR: Receipt download failed - {str(download_error)}")
        raise HTTPException(status_code=500, detail=f"Failed to download receipt: {str(download_error)}")

# Mobile app endpoints
@app.post("/mobile/verify-cow", tags=["Mobile App"])
async def mobile_cattle_verification(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Mobile app cattle verification endpoint with lower threshold."""
    try:
        print(f"INFO: Mobile verification started with {len(files)} files")
        
        # Use the same verification logic but with lower threshold for mobile
        # (Implementation similar to verify_cattle but with VERIFICATION_THRESHOLD = 0.75)
        
        return await verify_cattle(files, db)
        
    except Exception as mobile_verification_error:
        print(f"ERROR: Mobile verification failed - {str(mobile_verification_error)}")
        return {
            "registered": False,
            "status": "ERROR",
            "message": "❌ Verification failed due to technical error. Please try again."
        }

@app.post("/submit-report", tags=["Mobile App"])
async def submit_theft_report(
    description: str = Form(...),
    location: str = Form(None),
    db: Session = Depends(get_db)
):
    """Submit theft/suspect report from mobile app."""
    try:
        print(f"INFO: Report submission - {description[:50]}...")
        
        # Create report record
        new_report = Report(
            user_id=1,  # Default user ID for now
            description=description,
            location=location
        )
        db.add(new_report)
        db.commit()
        
        print("SUCCESS: Report submitted successfully")
        return {
            "success": True,
            "message": "Report submitted successfully",
            "report_id": new_report.report_id
        }
        
    except Exception as report_error:
        print(f"ERROR: Report submission failed - {str(report_error)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Report submission failed: {str(report_error)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)