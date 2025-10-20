# main.py - Titweng Cattle Recognition Backend (No Auth Version for Render)

import os
from datetime import datetime, timedelta
from typing import Optional, List, Annotated
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from pydantic import BaseModel, Field
from fastapi.openapi.utils import get_openapi
from typing import Annotated
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from contextlib import asynccontextmanager
from twilio.rest import Client
from reportlab.pdfgen import canvas
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database models and functions
from app.database import (
    Owner, Cow, User, Embedding, Report, VerificationLog,
    get_db, SessionLocal
)

# ================================
# NO AUTHENTICATION - OPEN ACCESS
# ================================
def get_current_user(db: Session = Depends(get_db)):
    return {"user_id": 1, "username": "admin", "role": "admin"}

def get_current_admin(db: Session = Depends(get_db)):
    return {"user_id": 1, "username": "admin", "role": "admin"}

# ================================
# ML MODEL
# ================================
class SiameseNetwork(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        backbone = models.resnet18(pretrained=False)
        modules = list(backbone.children())[:-1]
        self.backbone = torch.nn.Sequential(*modules)
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, embedding_dim)
        )

    def forward_one(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)

siamese_model = None

def load_siamese_model():
    global siamese_model
    model_path = "ml_models/siamese/siamese_model_final.pth"
    if not os.path.exists(model_path):
        print("Siamese model not found")
        return None
    try:
        model = SiameseNetwork(embedding_dim=256)
        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print("Siamese model loaded")
        return model
    except Exception as e:
        print(f"Model loading failed: {e}")
        return None

# ================================
# ML UTILITIES
# ================================
def preprocess_image(image: np.ndarray, size=(128,128)) -> torch.Tensor:
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(Image.fromarray(image)).unsqueeze(0)

def extract_embedding(image_tensor: torch.Tensor) -> np.ndarray:
    global siamese_model
    if siamese_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    with torch.no_grad():
        emb = siamese_model.forward_one(image_tensor)
        embedding = emb.cpu().numpy().flatten()
        
        # Quality validation for embedding
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 0.5 or embedding_norm > 2.0:
            raise ValueError("Poor quality embedding - image may be unclear")
            
        return embedding

def detect_nose(image: np.ndarray) -> Optional[np.ndarray]:
    """Process pre-cropped nose print images"""
    # Images are already cropped nose prints - return as is
    # Just validate image quality
    if image.size == 0:
        return None
    
    # Basic quality check
    variance = np.var(image)
    if variance < 50:  # Very low variance = likely blank/poor image
        return None
        
    return image

# ================================
# TWILIO CONFIGURATION
# ================================
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN else None

# ================================
# NOTIFICATIONS
# ================================
def send_sms(phone_number: str, message: str):
    try:
        if twilio_client and TWILIO_PHONE_NUMBER:
            twilio_client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            print(f"SMS sent to {phone_number}")
        else:
            print(f"SMS would be sent to {phone_number}: {message}")
    except Exception as e:
        print(f"SMS failed: {e}")

def generate_pdf_receipt(cow_tag: str, owner_name: str, cow_data: dict = None) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import HexColor, black, white
    from reportlab.lib.units import mm
    import qrcode
    import json
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Background
    c.setFillColor(HexColor('#f8f9fa'))
    c.rect(0, 0, width, height, fill=1)
    
    # Header
    c.setFillColor(HexColor('#2c3e50'))
    c.rect(0, height-60*mm, width, 60*mm, fill=1)
    
    # Logo/Title
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    title_text = "TITWENG CATTLE REGISTRATION"
    text_width = c.stringWidth(title_text, "Helvetica-Bold", 20)
    c.drawString((width - text_width) / 2, height-35*mm, title_text)
    
    c.setFont("Helvetica-Bold", 14)
    subtitle_text = "OFFICIAL REGISTRATION RECEIPT"
    text_width = c.stringWidth(subtitle_text, "Helvetica-Bold", 14)
    c.drawString((width - text_width) / 2, height-45*mm, subtitle_text)
    
    # Main content area
    c.setFillColor(black)
    y_pos = height - 80*mm
    
    # Cow Information Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y_pos, "CATTLE INFORMATION")
    c.setLineWidth(1)
    c.line(20*mm, y_pos-2*mm, 190*mm, y_pos-2*mm)
    y_pos -= 10*mm
    
    c.setFont("Helvetica", 11)
    c.drawString(20*mm, y_pos, f"Cow Tag: {cow_tag}")
    y_pos -= 8*mm
    
    if cow_data:
        c.drawString(20*mm, y_pos, f"Breed: {cow_data.get('breed', 'Not specified')}")
        y_pos -= 6*mm
        c.drawString(20*mm, y_pos, f"Color: {cow_data.get('color', 'Not specified')}")
        y_pos -= 6*mm
        c.drawString(20*mm, y_pos, f"Age: {cow_data.get('age', 'Not specified')} years" if cow_data.get('age') else "Age: Not specified")
        y_pos -= 6*mm
        c.drawString(20*mm, y_pos, f"Registration Date: {cow_data.get('registration_date', datetime.now().strftime('%d/%m/%Y %H:%M'))}")
        y_pos -= 10*mm
    
    # Owner Information Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y_pos, "OWNER INFORMATION")
    c.line(20*mm, y_pos-2*mm, 190*mm, y_pos-2*mm)
    y_pos -= 10*mm
    
    c.setFont("Helvetica", 11)
    c.drawString(20*mm, y_pos, f"Full Name: {owner_name}")
    y_pos -= 6*mm
    
    if cow_data and cow_data.get('owner_data'):
        owner_data = cow_data['owner_data']
        c.drawString(20*mm, y_pos, f"Email: {owner_data.get('email', 'Not provided')}")
        y_pos -= 6*mm
        c.drawString(20*mm, y_pos, f"Phone: {owner_data.get('phone', 'Not provided')}")
        y_pos -= 6*mm
        c.drawString(20*mm, y_pos, f"Address: {owner_data.get('address', 'Not provided')}")
        y_pos -= 6*mm
        if owner_data.get('national_id'):
            c.drawString(20*mm, y_pos, f"National ID: {owner_data.get('national_id')}")
            y_pos -= 10*mm
    
    # System Information
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y_pos, "SYSTEM INFORMATION")
    c.line(20*mm, y_pos-2*mm, 190*mm, y_pos-2*mm)
    y_pos -= 10*mm
    
    c.setFont("Helvetica", 11)
    c.drawString(20*mm, y_pos, f"Receipt Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    y_pos -= 6*mm
    c.drawString(20*mm, y_pos, f"System: Titweng Cattle Recognition System")
    y_pos -= 6*mm
    c.drawString(20*mm, y_pos, f"Status: Officially Registered")
    
    # Generate comprehensive QR code data
    qr_data = {
        "system": "Titweng",
        "cow_tag": cow_tag,
        "owner_name": owner_name,
        "registration_date": datetime.now().strftime('%Y-%m-%d'),
        "verification_url": f"https://titweng.com/verify/{cow_tag}"
    }
    
    if cow_data:
        qr_data.update({
            "breed": cow_data.get('breed', ''),
            "color": cow_data.get('color', ''),
            "age": cow_data.get('age', '')
        })
    
    # Create QR code with JSON data
    qr_json = json.dumps(qr_data, separators=(',', ':'))
    qr = qrcode.QRCode(version=3, box_size=4, border=2)
    qr.add_data(qr_json)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Save QR temporarily and add to PDF
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as qr_temp:
        qr_img.save(qr_temp.name, format='PNG')
        qr_temp_path = qr_temp.name
    
    # Position QR code
    qr_x = width - 50*mm
    qr_y = height - 160*mm
    c.drawImage(qr_temp_path, qr_x, qr_y, width=40*mm, height=40*mm)
    os.unlink(qr_temp_path)
    
    # QR Code label
    c.setFont("Helvetica-Bold", 10)
    c.drawString(qr_x, qr_y-8*mm, "Scan for Verification")
    
    # Footer
    c.setFont("Helvetica", 8)
    footer_text = "This is an official registration receipt. Keep it safe for verification purposes."
    c.drawString(20*mm, 20*mm, footer_text)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

def send_email_with_receipt(owner_email: str, owner_name: str, cow_tag: str, pdf_receipt: bytes):
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not all([smtp_server, smtp_username, smtp_password]):
            print("Email configuration missing")
            return
        
        print(f"Attempting to send email to {owner_email} using {smtp_username}")
        print(f"SMTP Config: Server={smtp_server}, Port={smtp_port}, User={smtp_username}")
        
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = owner_email
        msg['Subject'] = f"Cattle Registration Confirmation - {cow_tag}"
        
        body = f"""Dear {owner_name},

Your cattle has been successfully registered with tag: {cow_tag}

Please find your registration receipt attached.

Best regards,
Titweng Cattle Recognition System"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_receipt)
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{cow_tag}_receipt.pdf"')
        msg.attach(part)
        
        # Send email with better error handling
        print("Connecting to SMTP server...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        print("Starting TLS...")
        server.starttls()
        print("Logging in...")
        server.login(smtp_username, smtp_password)
        print("Sending message...")
        text = msg.as_string()
        server.sendmail(smtp_username, owner_email, text)
        server.quit()
        print("Email sent successfully!")
        
        print(f"Email successfully sent to {owner_email}")
    except Exception as e:
        print(f"Email failed: {str(e)}")
        print(f"SMTP Config: {os.getenv('SMTP_SERVER')}:{os.getenv('SMTP_PORT')} User: {os.getenv('SMTP_USERNAME')}")

def generate_next_cow_tag(db: Session) -> str:
    """Generate secure cow tag with random component"""
    import random
    import string
    
    # Get count for base number
    result = db.execute(text("SELECT COUNT(*) FROM cows"))
    total_cows = result.fetchone()[0]
    
    # Generate secure tag: TW + 2 random letters + 4 digits
    random_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
    sequential_number = f"{total_cows + 1:04d}"
    
    secure_tag = f"TW{random_letters}{sequential_number}"
    
    # Ensure uniqueness
    while db.query(Cow).filter(Cow.cow_tag == secure_tag).first():
        random_letters = ''.join(random.choices(string.ascii_uppercase, k=2))
        secure_tag = f"TW{random_letters}{sequential_number}"
    
    return secure_tag

# ================================
# FASTAPI APP
# ================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global siamese_model
    os.makedirs("static", exist_ok=True)
    siamese_model = load_siamese_model()
    
    # Create database tables
    try:
        from app.database import Base, engine
        
        # Create pgvector extension
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("pgvector extension created/verified")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
            
    except Exception as e:
        print(f"Error creating tables: {e}")
    
    yield
    print("Shutting down...")

app = FastAPI(title="Titweng Cattle Recognition API (No Auth)", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ================================
# API ENDPOINTS
# ================================

@app.get("/", tags=["Status"])
def root():
    return {"message": "Titweng Cattle Recognition API - No Authentication Required", "status": "live"}

@app.post("/test-simple", tags=["Status"])
def test_simple(test_data: str = Form(...), db: Session = Depends(get_db)):
    """Simple test endpoint without file processing"""
    try:
        # Test database
        result = db.execute(text("SELECT COUNT(*) FROM cows"))
        cow_count = result.fetchone()[0]
        
        return {
            "success": True,
            "test_data": test_data,
            "cow_count": cow_count,
            "message": "Database and basic functionality working"
        }
    except Exception as e:
        print(f"Simple test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

# -------- Admin Dashboard --------
@app.post("/register-cow", tags=["Admin Dashboard"])
async def register_cow(
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
    try:
        print(f"Registration started for {owner_name}")
        
        # Auto-generate secure cow_tag
        cow_tag = generate_next_cow_tag(db)
        
        # Handle age conversion
        cow_age = None
        if age and age.isdigit():
            cow_age = int(age)
        
        # Create or get owner
        owner = db.query(Owner).filter(Owner.email == owner_email).first()
        if not owner:
            owner = Owner(full_name=owner_name, email=owner_email, phone=owner_phone, address=owner_address, national_id=national_id)
            db.add(owner)
            db.commit()
            db.refresh(owner)
        
        embeddings_list = []
        
        # Create directory for cow images
        os.makedirs("static/cow_images", exist_ok=True)
        
        for i, f in enumerate(files[:5]):
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                continue
            nose = detect_nose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if nose is None:
                continue
            try:
                tensor = preprocess_image(nose)
                emb = extract_embedding(tensor)
                
                # Convert image to bytes for database storage
                _, buffer = cv2.imencode('.jpg', image)
                image_bytes = buffer.tobytes()
                
                embeddings_list.append((emb, image_bytes))
            except ValueError as e:
                print(f"Skipping poor quality image: {e}")
                continue
        
        if len(embeddings_list) < 1:
            raise HTTPException(status_code=400, detail="Need at least 1 high-quality nose print image for registration. Please ensure images are clear and well-lit.")
        
        print(f"Processed {len(embeddings_list)} embeddings successfully")
        
        # Create cow and embeddings
        # Create cow and embeddings
        cow = Cow(cow_tag=cow_tag, breed=breed, color=color, age=cow_age, owner_id=owner.owner_id)
        db.add(cow)
        db.flush()
        
        # Add embeddings
        for i, (emb, img_bytes) in enumerate(embeddings_list):
            image_filename = f"{cow_tag}_{i+1:03d}.jpg"
            
            embedding = Embedding(
                cow_id=cow.cow_id, 
                embedding=emb.tolist(),
                image_path=image_filename,
                image_data=img_bytes
            )
            db.add(embedding)
        
        db.commit()
        
        # Generate PDF receipt with complete cow data
        cow_data = {
            'breed': breed,
            'color': color,
            'age': cow_age,
            'registration_date': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'owner_data': {
                'email': owner_email,
                'phone': owner_phone,
                'address': owner_address,
                'national_id': national_id
            }
        }
        pdf_receipt = generate_pdf_receipt(cow_tag, owner_name, cow_data)
        
        # Save receipt
        os.makedirs("static/receipts", exist_ok=True)
        receipt_path = f"static/receipts/{cow_tag}_receipt.pdf"
        with open(receipt_path, "wb") as f:
            f.write(pdf_receipt)
        
        # Update cow with receipt link
        cow.receipt_pdf_link = receipt_path
        qr_data = f"COW:{cow_tag}|OWNER:{owner_name}|DATE:{datetime.now().strftime('%Y%m%d')}"
        cow.qr_code_link = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={qr_data}"
        db.commit()
        
        # Send notifications
        try:
            print("Sending SMS notification...")
            send_sms(owner_phone, f"Your cow {cow_tag} has been registered successfully!")
            print("Sending email notification...")
            send_email_with_receipt(owner_email, owner_name, cow_tag, pdf_receipt)
            print("Notifications sent successfully")
        except Exception as e:
            print(f"Notification error: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return {
            "success": True, 
            "cow_id": cow.cow_id,
            "cow_tag": cow_tag,
            "receipt_path": receipt_path,
            "message": f"Cow registered successfully with tag {cow_tag}",
            "notifications": "SMS and email sent"
        }
        
    except Exception as e:
        print(f"Registration error: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

# -------- Admin Verification --------
@app.post("/verify-cow", tags=["Admin Dashboard"])
async def verify_cow(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    try:
        print(f"Verification started with {len(files)} files")
        
        if not files:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Process images
        query_embeddings = []
        for f in files[:3]:
            contents = await f.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                continue
            nose = detect_nose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if nose is None:
                continue
            try:
                tensor = preprocess_image(nose)
                emb = extract_embedding(tensor)
                query_embeddings.append(emb)
            except ValueError:
                continue
        
        if len(query_embeddings) == 0:
            raise HTTPException(status_code=400, detail="No valid images for verification")
        
        # Use average of query embeddings
        query_emb = np.mean(query_embeddings, axis=0)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Check against database
        all_cows = db.query(Cow).filter(Cow.embeddings.any()).all()
        
        if not all_cows:
            return {
                "registered": False, 
                "status": "NOT_REGISTERED",
                "message": "❌ COW NOT REGISTERED - No cows in database"
            }
        cow_scores = []
        for cow in all_cows:
            if not cow.embeddings or len(cow.embeddings) == 0:
                continue
                
            similarities = []
            for stored_emb in cow.embeddings:
                stored_vector = np.array(stored_emb.embedding, dtype=np.float32)
                stored_norm = stored_vector / np.linalg.norm(stored_vector)
                cosine_sim = np.dot(query_emb, stored_norm)
                similarities.append(cosine_sim)
            
            similarities = np.array(similarities)
            top3_avg = np.mean(np.sort(similarities)[-3:])
            cow_scores.append((cow, top3_avg))
        
        # Sort by score
        cow_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_match = cow_scores[0][0]
        best_score = cow_scores[0][1]
        
        VERIFICATION_THRESHOLD = 0.82
        
        if best_score >= VERIFICATION_THRESHOLD:
            owner = best_match.owner
            
            # Log verification
            verification_log = VerificationLog(
                cow_id=best_match.cow_id,
                similarity_score=float(best_score),  # Convert numpy.float32 to Python float
                verified=True,
                user_id=1
            )
            db.add(verification_log)
            db.commit()
            
            # Send verification SMS
            if owner:
                send_sms(owner.phone, f"Your cow {best_match.cow_tag} was successfully verified!")
            
            return {
                "registered": True,
                "status": "REGISTERED",
                "message": "✅ COW IS REGISTERED - Verification Successful",
                "cow_details": {
                    "cow_tag": best_match.cow_tag,
                    "cow_id": best_match.cow_id,
                    "breed": best_match.breed or "Not specified",
                    "color": best_match.color or "Not specified",
                    "age": f"{best_match.age} years" if best_match.age else "Not specified"
                },
                "owner_details": {
                    "owner_name": owner.full_name if owner else "Unknown",
                    "owner_email": owner.email if owner else "Unknown",
                    "owner_phone": owner.phone if owner else "Unknown"
                },
                "verification_info": {
                    "similarity_score": round(float(best_score), 4),
                    "confidence_level": "HIGH" if best_score > 0.85 else "MEDIUM"
                }
            }
        else:
            # Log failed verification
            verification_log = VerificationLog(
                similarity_score=float(best_score),  # Convert numpy.float32 to Python float
                verified=False,
                user_id=1
            )
            db.add(verification_log)
            db.commit()
            
            return {
                "registered": False,
                "status": "NOT_REGISTERED", 
                "message": "❌ COW NOT REGISTERED - This cow is not in the system",
                "verification_info": {
                    "highest_similarity": round(float(best_score), 4),
                    "threshold_required": VERIFICATION_THRESHOLD
                }
            }
    except Exception as e:
        print(f"Verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

@app.get("/health", tags=["Status"])
def health_check():
    try:
        # Test database connection
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_connected = True
    except Exception as e:
        print(f"Database connection error: {e}")
        db_connected = False
    
    return {
        "status": "healthy" if db_connected else "database_error",
        "model_loaded": siamese_model is not None,
        "database_connected": db_connected,
        "auth": "disabled",
        "timestamp": datetime.now().isoformat()
    }



@app.get("/admin/dashboard", tags=["Admin Dashboard"])
def admin_dashboard(db: Session = Depends(get_db)):
    """Admin dashboard statistics"""
    total_cows = db.query(Cow).count()
    total_owners = db.query(Owner).count()
    pending_reports = db.query(Report).filter(Report.status == "pending").count()
    return {"total_cows": total_cows, "total_owners": total_owners, "pending_reports": pending_reports}

@app.get("/admin/reports", tags=["Admin Dashboard"])
def get_reports(db: Session = Depends(get_db)):
    """Get all reports for admin review"""
    reports = db.query(Report).order_by(Report.created_at.desc()).all()
    return [{"report_id": r.report_id, "description": r.description, "location": r.location,
             "status": r.status, "created_at": r.created_at} for r in reports]

@app.get("/admin/cows", tags=["Admin Dashboard"])
def get_all_cows(db: Session = Depends(get_db)):
    """View all registered cows with owner details"""
    cows = db.query(Cow).all()
    result = []
    for cow in cows:
        owner = cow.owner
        result.append({
            "cow_id": cow.cow_id,
            "cow_tag": cow.cow_tag,
            "breed": cow.breed,
            "color": cow.color,
            "age": cow.age,
            "registration_date": cow.registration_date.strftime('%d/%m/%Y %H:%M') if cow.registration_date else "Unknown",
            "owner_name": owner.full_name if owner else "Unknown",
            "owner_email": owner.email if owner else "Unknown",
            "owner_phone": owner.phone if owner else "Unknown",
            "owner_address": owner.address if owner else "Unknown",
            "embeddings_count": len(cow.embeddings)
        })
    return {"total_cows": len(result), "cows": result}

@app.put("/admin/cows/{cow_tag}", tags=["Admin Dashboard"])
def update_cow(
    cow_tag: str,
    owner_name: str = Form(None),
    owner_email: str = Form(None),
    owner_phone: str = Form(None),
    owner_address: str = Form(None),
    national_id: str = Form(None),
    breed: str = Form(None),
    color: str = Form(None),
    age: int = Form(None),
    db: Session = Depends(get_db)
):
    """Update cow and owner information - cow_tag remains immutable for security"""
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Update cow details
    if breed is not None:
        cow.breed = breed
    if color is not None:
        cow.color = color
    if age is not None:
        cow.age = age
    
    # Update owner details
    if cow.owner and any([owner_name, owner_email, owner_phone, owner_address, national_id]):
        if owner_name is not None:
            cow.owner.full_name = owner_name
        if owner_email is not None:
            cow.owner.email = owner_email
        if owner_phone is not None:
            cow.owner.phone = owner_phone
        if owner_address is not None:
            cow.owner.address = owner_address
        if national_id is not None:
            cow.owner.national_id = national_id
    
    db.commit()
    
    return {
        "success": True,
        "message": f"Cow {cow_tag} updated successfully",
        "cow_id": cow.cow_id,
        "cow_tag": cow.cow_tag,
        "note": "cow_tag remains unchanged for security and traceability"
    }

@app.delete("/admin/cows/{cow_tag}", tags=["Admin Dashboard"])
def delete_cow(
    cow_tag: str,
    db: Session = Depends(get_db)
):
    """Delete cow and all associated data using cow_tag"""
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    cow_tag = cow.cow_tag
    owner_name = cow.owner.full_name if cow.owner else "Unknown"
    
    # Delete associated files
    try:
        if cow.receipt_pdf_link and os.path.exists(cow.receipt_pdf_link):
            os.remove(cow.receipt_pdf_link)
        
        # Delete cow images from filesystem
        for embedding in cow.embeddings:
            if embedding.image_path:
                image_path = os.path.join("static/cow_images", embedding.image_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
    except Exception as e:
        print(f"File cleanup warning: {e}")
    
    # Delete verification logs first (foreign key constraint)
    db.query(VerificationLog).filter(VerificationLog.cow_id == cow.cow_id).delete()
    
    # Database will cascade delete embeddings
    db.delete(cow)
    db.commit()
    
    return {
        "success": True,
        "message": f"Cow {cow_tag} (Owner: {owner_name}) deleted successfully",
        "deleted_cow_tag": cow_tag
    }

# -------- Mobile App --------
@app.post("/mobile/verify-cow", tags=["Mobile App"])
async def mobile_verify_cow(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Mobile app cow verification endpoint"""
    try:
        print(f"Mobile verification started with {len(files)} files")
        
        if not files:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Process images
        query_embeddings = []
        for f in files[:3]:
            try:
                contents = await f.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                nose = detect_nose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if nose is None:
                    continue
                tensor = preprocess_image(nose)
                emb = extract_embedding(tensor)
                query_embeddings.append(emb)
            except Exception as e:
                print(f"Image processing error: {e}")
                continue
        
        if len(query_embeddings) == 0:
            return {
                "registered": False,
                "status": "ERROR",
                "message": "❌ No valid images for verification. Please upload clear nose print images."
            }
        
        # Use average of query embeddings
        query_emb = np.mean(query_embeddings, axis=0)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Check against database
        all_cows = db.query(Cow).filter(Cow.embeddings.any()).all()
        
        if not all_cows:
            return {
                "registered": False, 
                "status": "NOT_REGISTERED",
                "message": "❌ COW NOT REGISTERED - No cows in database"
            }
        
        cow_scores = []
        for cow in all_cows:
            if not cow.embeddings or len(cow.embeddings) == 0:
                continue
                
            similarities = []
            for stored_emb in cow.embeddings:
                try:
                    stored_vector = np.array(stored_emb.embedding, dtype=np.float32)
                    stored_norm = stored_vector / np.linalg.norm(stored_vector)
                    cosine_sim = np.dot(query_emb, stored_norm)
                    similarities.append(cosine_sim)
                except Exception as e:
                    print(f"Similarity calculation error: {e}")
                    continue
            
            if similarities:
                similarities = np.array(similarities)
                avg_similarity = np.mean(similarities)
                cow_scores.append((cow, avg_similarity))
        
        if not cow_scores:
            return {
                "registered": False,
                "status": "ERROR",
                "message": "❌ Unable to process verification. Please try again."
            }
        
        # Sort by score
        cow_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_match = cow_scores[0][0]
        best_score = cow_scores[0][1]
        
        VERIFICATION_THRESHOLD = 0.75  # Lower threshold for mobile
        
        if best_score >= VERIFICATION_THRESHOLD:
            owner = best_match.owner
            
            # Log verification
            verification_log = VerificationLog(
                cow_id=best_match.cow_id,
                similarity_score=float(best_score),
                verified=True,
                user_id=1
            )
            db.add(verification_log)
            db.commit()
            
            # Send verification SMS
            if owner and owner.phone:
                try:
                    send_sms(owner.phone, f"Your cow {best_match.cow_tag} was successfully verified via mobile app!")
                except Exception as e:
                    print(f"SMS sending error: {e}")
            
            return {
                "registered": True,
                "status": "REGISTERED",
                "message": "✅ COW IS REGISTERED - Verification Successful",
                "cow_details": {
                    "cow_tag": best_match.cow_tag,
                    "breed": best_match.breed or "Not specified",
                    "color": best_match.color or "Not specified",
                    "age": f"{best_match.age} years" if best_match.age else "Not specified"
                },
                "owner_details": {
                    "owner_name": owner.full_name if owner else "Unknown",
                    "owner_phone": owner.phone if owner else "Unknown"
                },
                "verification_info": {
                    "similarity_score": round(float(best_score), 4),
                    "confidence_level": "HIGH" if best_score > 0.85 else "MEDIUM"
                }
            }
        else:
            # Log failed verification
            verification_log = VerificationLog(
                similarity_score=float(best_score),
                verified=False,
                user_id=1
            )
            db.add(verification_log)
            db.commit()
            
            return {
                "registered": False,
                "status": "NOT_REGISTERED", 
                "message": "❌ COW NOT REGISTERED - This cow is not in the system",
                "verification_info": {
                    "highest_similarity": round(float(best_score), 4),
                    "threshold_required": VERIFICATION_THRESHOLD
                }
            }
    except Exception as e:
        print(f"Mobile verification error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "registered": False,
            "status": "ERROR",
            "message": "❌ Verification failed due to technical error. Please try again."
        }

@app.post("/submit-report", tags=["Mobile App"])
async def submit_report(
    description: str = Form(...),
    location: str = Form(None),
    db: Session = Depends(get_db)
):
    try:
        print(f"Report submission started: {description[:50]}...")
        
        # Submit a report from mobile app
        report = Report(user_id=1, description=description, location=location)  # Dummy user ID
        db.add(report)
        db.commit()
        print("Report submitted successfully")
        return {"success": True, "message": "Report submitted successfully"}
    except Exception as e:
        print(f"Report submission error: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Report submission failed: {str(e)}")

@app.get("/preview-next-cow-tag", tags=["Admin Dashboard"])
def preview_next_cow_tag(db: Session = Depends(get_db)):
    """Preview what the next cow tag will be"""
    next_tag = generate_next_cow_tag(db)
    total_cows = db.query(Cow).count()
    return {
        "next_cow_tag": next_tag,
        "explanation": "This tag will be automatically generated during registration",
        "current_total_cows": total_cows,
        "note": "cow_id and cow_tag are auto-generated for security"
    }

@app.get("/image/{embedding_id}", tags=["Utility"])
def get_image(embedding_id: int, db: Session = Depends(get_db)):
    """Get image from database by embedding ID"""
    embedding = db.query(Embedding).filter(Embedding.embedding_id == embedding_id).first()
    if not embedding or not embedding.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return Response(
        content=embedding.image_data,
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename={embedding.image_path}"}
    )

@app.get("/download-receipt/{cow_tag}", tags=["Utility"])
def download_receipt(cow_tag: str, db: Session = Depends(get_db)):
    """Download PDF receipt for a registered cow"""
    try:
        cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
        if not cow:
            raise HTTPException(status_code=404, detail="Cow not found")
        
        # Check if receipt exists in database
        if not cow.receipt_pdf_link:
            # Generate receipt if it doesn't exist
            owner_name = cow.owner.full_name if cow.owner else "Unknown Owner"
            cow_data = {
                'breed': cow.breed,
                'color': cow.color,
                'age': cow.age,
                'registration_date': cow.registration_date.strftime('%d/%m/%Y %H:%M') if cow.registration_date else 'Unknown',
                'owner_data': {
                    'email': cow.owner.email if cow.owner else '',
                    'phone': cow.owner.phone if cow.owner else '',
                    'address': cow.owner.address if cow.owner else '',
                    'national_id': cow.owner.national_id if cow.owner else ''
                }
            }
            pdf_receipt = generate_pdf_receipt(cow_tag, owner_name, cow_data)
            
            # Save receipt
            os.makedirs("static/receipts", exist_ok=True)
            receipt_path = f"static/receipts/{cow_tag}_receipt.pdf"
            with open(receipt_path, "wb") as f:
                f.write(pdf_receipt)
            
            # Update cow record
            cow.receipt_pdf_link = receipt_path
            db.commit()
        
        # Check if file exists on filesystem
        if not os.path.exists(cow.receipt_pdf_link):
            # Regenerate if file is missing
            owner_name = cow.owner.full_name if cow.owner else "Unknown Owner"
            cow_data = {
                'breed': cow.breed,
                'color': cow.color,
                'age': cow.age,
                'registration_date': cow.registration_date.strftime('%d/%m/%Y %H:%M') if cow.registration_date else 'Unknown',
                'owner_data': {
                    'email': cow.owner.email if cow.owner else '',
                    'phone': cow.owner.phone if cow.owner else '',
                    'address': cow.owner.address if cow.owner else '',
                    'national_id': cow.owner.national_id if cow.owner else ''
                }
            }
            pdf_receipt = generate_pdf_receipt(cow_tag, owner_name, cow_data)
            
            os.makedirs("static/receipts", exist_ok=True)
            with open(cow.receipt_pdf_link, "wb") as f:
                f.write(pdf_receipt)
        
        return FileResponse(
            path=cow.receipt_pdf_link,
            filename=f"{cow_tag}_receipt.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        print(f"Receipt download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download receipt: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)