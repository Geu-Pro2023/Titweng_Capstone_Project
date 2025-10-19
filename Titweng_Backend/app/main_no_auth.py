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

@app.post("/register-cow", tags=["Cattle Management"])
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
    
    if len(embeddings_list) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 high-quality images for registration")
    
    try:
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
        
        return {
            "success": True, 
            "cow_id": cow.cow_id,
            "cow_tag": cow_tag,
            "message": f"Cow registered successfully with tag {cow_tag}"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/verify-cow", tags=["Cattle Management"])
async def verify_cow(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
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
        if not cow.embeddings or len(cow.embeddings) < 3:
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
            similarity_score=best_score,
            verified=True,
            user_id=1
        )
        db.add(verification_log)
        db.commit()
        
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
                "similarity_score": round(best_score, 4),
                "confidence_level": "HIGH" if best_score > 0.85 else "MEDIUM"
            }
        }
    else:
        # Log failed verification
        verification_log = VerificationLog(
            similarity_score=best_score,
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
                "highest_similarity": round(best_score, 4),
                "threshold_required": VERIFICATION_THRESHOLD
            }
        }

@app.get("/health", tags=["Status"])
def health_check():
    return {"status": "healthy", "model_loaded": siamese_model is not None, "auth": "disabled"}

@app.get("/cows", tags=["Cattle Management"])
def get_all_cows(db: Session = Depends(get_db)):
    """View all registered cows"""
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
            "owner_name": owner.full_name if owner else "Unknown",
            "owner_email": owner.email if owner else "Unknown",
            "embeddings_count": len(cow.embeddings)
        })
    return {"total_cows": len(result), "cows": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)