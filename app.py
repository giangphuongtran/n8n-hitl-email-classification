"""
FastAPI Email Classification Service - PRODUCTION VERSION
Hybrid approach: Fine-tuned DistilBERT + Zero-shot mDeBERTa fallback
Supports human feedback loop for continuous improvement
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from datetime import datetime

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)

# ========================
# Logging Setup
# ========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================
# FastAPI App
# ========================
app = FastAPI(
    title="Email Classification API",
    description="Hybrid fine-tuned + zero-shot email classifier with human feedback loop",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Configuration
# ========================
FINETUNED_MODEL_PATH = "./email_classifier_finetuned"
CONFIDENCE_THRESHOLD = 0.85
CONFIDENCE_THRESHOLD_AUTO = 0.75

# All categories (11 categories for 3000+ samples)
CATEGORIES = [
    "promotional offers, discounts, and marketing emails",
    "job recruitment, career opportunities, and professional networking",
    "account security alerts, login notifications, and password resets",
    "academic coursework, university, and school communications",
    "personal direct messages and conversations",
    "spam, scam, and unsolicited junk mail",
    "order confirmations, receipts, and transactional notifications",
    "newsletters, blogs, and news digests",
    "travel bookings, flight itineraries, and hotel reservations",
    "banking, finance, and payment notifications",
    "social media notifications and platform updates",
]

# ========================
# Model Loading
# ========================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Try to load fine-tuned model
finetuned_model = None
finetuned_tokenizer = None
id2label = None
USE_FINETUNED = False

try:
    logger.info(f"Loading fine-tuned model from {FINETUNED_MODEL_PATH}...")
    finetuned_tokenizer = DistilBertTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    finetuned_model = DistilBertForSequenceClassification.from_pretrained(
        FINETUNED_MODEL_PATH
    ).to(device)
    
    # Load label mapping
    label_mapping_path = Path(FINETUNED_MODEL_PATH) / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path) as f:
            label_mapping = json.load(f)
            id2label = {int(k): v for k, v in label_mapping['id2label'].items()}
            
    finetuned_model.eval()
    USE_FINETUNED = True
    logger.info("✓ Fine-tuned model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️  Could not load fine-tuned model: {e}")
    logger.info("Falling back to zero-shot classification only")
    
# Load zero-shot model
zero_shot_classifier = None
try:
    logger.info("Loading zero-shot classifier (mDeBERTa)...")
    zero_shot_classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        device=0 if device == "cuda" else -1
    )
    logger.info("✓ Zero-shot classifier loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load zero-shot classifier: {e}")
    zero_shot_classifier = None
    
# ========================
# Pydantic Models
# ========================
class EmailRequest(BaseModel):
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    email_id: Optional[str] = Field(None, description="Unique email identifier for feedback loop")
    sender: Optional[str] = Field(None, description="Email sender address")
    
class ClassificationResponse(BaseModel):
    email_id: Optional[str] = None
    category: str
    confidence: float
    all_scores: Dict[str, float]
    model_used: str
    requires_review: bool
    confidence_threshold: float
    timestamp: str
    
class FeedbackRequest(BaseModel):
    email_id: str
    correct_category: str
    incorrect_category: Optional[str] = None
    feedback_type: str = Field("correction", description="'correction' or 'confirmation'")
    notes: Optional[str] = None
    
class HealthResponse(BaseModel):
    status: str
    finetuned_available: bool
    zero_shot_available: bool
    active_model: str
    device: str
    
# ========================
# Helper Functions
# ========================
def prepare_text(subject: str, body: str, max_body_length: int = 500) -> str:
    """Prepare email text for classification."""
    return f"Subject: {subject} Body: {body[:max_body_length]}"

def classify_finetuned(text: str) -> tuple:
    """Classify using fine-tuned model."""
    try:
        inputs = finetuned_tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = finetuned_model(**inputs)
        
        # Get probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        
        # Get predictions
        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].cpu())
        
        # Build all scores
        all_scores = {}
        if id2label:
            for idx, label in id2label.items():
                all_scores[label] = round(float(probs[idx].cpu()), 4)
        else:
            for i, cat in enumerate(CATEGORIES):
                all_scores[cat] = round(float(probs[min(i, len(probs)-1)].cpu()), 4)
                
        predicted_category = id2label.get(pred_idx, CATEGORIES[pred_idx])
        
        return predicted_category, confidence, all_scores, "fine-tuned"
    
    except Exception as e:
        logger.error(f"Error in fine-tuned inference: {e}")
        raise
    
def classify_zero_shot(text: str) -> tuple:
    """Classify using zero-shot model."""
    try:
        result = zero_shot_classifier(
            text,
            CATEGORIES,
            hypothesis_template="This email is about {}.",
        )
        
        all_scores = {cat: round(score, 4)
                      for cat, score in zip(result['labels'], result['scores'])}
        
        return result['labels'][0], float(result['scores'][0]), all_scores, "zero-shot"
    except Exception as e:
        logger.error(f"Error in zero-shot inference: {e}")
        raise
    
# ========================
# API Endpoints
# ========================

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",  # FIX: Fixed typo (was 'statsus')
        finetuned_available=USE_FINETUNED,
        zero_shot_available=zero_shot_classifier is not None,
        active_model="fine-tuned" if USE_FINETUNED else "zero-shot",
        device=device,
    )
    
@app.post("/classify", response_model=ClassificationResponse)
def classify_email(request: EmailRequest):
    """
    Classify email with hybrid approach.
    
    Returns:
        - category: Predicted category
        - confidence: Confidence score (0-1)
        - all_scores: Scores for all categories
        - requires_review: True if confidence < threshold
        - model_used: "fine-tuned" or "zero-shot"
    """
    
    # Prepare email text
    text = prepare_text(request.subject, request.body)
    
    # Try fine-tuned classification first
    if USE_FINETUNED:   
        try:
            category, confidence, all_scores, model_used = classify_finetuned(text)
        except Exception as e:
            logger.warning(f"Fine-tuned classification failed, falling back to zero-shot: {e}")
            if zero_shot_classifier:
                category, confidence, all_scores, model_used = classify_zero_shot(text)
            else:
                raise HTTPException(status_code=500, detail="All classification methods failed")
    elif zero_shot_classifier:
        category, confidence, all_scores, model_used = classify_zero_shot(text)
    else:
        raise HTTPException(status_code=503, detail="No classification models available")
    
    # Determine if human review is required
    requires_review = confidence < CONFIDENCE_THRESHOLD_AUTO
    
    response = ClassificationResponse(
        email_id=request.email_id,
        category=category,
        confidence=round(confidence, 4),
        all_scores=all_scores,
        model_used=model_used,
        requires_review=requires_review,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        timestamp=datetime.utcnow().isoformat()
    )
    
    logger.info(
        f"Email {request.email_id}: {category} ({confidence:.2%}) "
        f"[{model_used}] {'→ REVIEW' if requires_review else '✓ AUTO'}"
    )
    
    return response

@app.post("/batch-classify")
def batch_classify(requests: List[EmailRequest]):
    """
    Classify multiple emails in batch.
    
    Useful for batch processing of archived emails.
    """
    results = []
    for req in requests:
        try:
            result = classify_email(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Error classifying email {req.email_id}: {e}")
            results.append({
                "email_id": req.email_id,
                "error": str(e)
            })
    return {
        "total": len(requests), 
        "successful": len([r for r in results if "error" not in r]),
        "results": results
    }
    
@app.post("/feedback")
def record_feedback(feedback: FeedbackRequest):
    """
    Record human feedback for model improvement.
    
    This stores corrections for batch retraining. In production,
    save to database for periodic fine-tuning.
    """
    feedback_data = {
        "email_id": feedback.email_id,
        "correct_category": feedback.correct_category,
        "incorrect_category": feedback.incorrect_category,
        "feedback_type": feedback.feedback_type,
        "notes": feedback.notes,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Save to feedback log (FIX: Use json.dumps not json.dump)
    feedback_file = Path("./feedback_log.jsonl")
    with open(feedback_file, "a") as f:
        f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        
    logger.info(f"Feedback recorded: {feedback.email_id} → {feedback.correct_category}")
    
    return {
        "status": "recorded",
        "feedback_id": feedback.email_id,
        "correct_category": feedback.correct_category
    }
    
@app.get("/stats")
def get_stats():
    """Get classification statistics."""
    feedback_file = Path("./feedback_log.jsonl")
    
    if not feedback_file.exists():
        return {
            "total_feedback": 0,
            "by_type": {}
        }
        
    feedback_by_type = {}
    with open(feedback_file) as f:
        for line in f:
            data = json.loads(line)
            ftype = data.get("feedback_type", "unknown")
            feedback_by_type[ftype] = feedback_by_type.get(ftype, 0) + 1
            
    return {
        "total_feedback": sum(feedback_by_type.values()),
        "by_type": feedback_by_type
    }
    
@app.get("/categories")
def get_categories():
    """Get list of available categories."""
    return {
        "categories": CATEGORIES,
        "count": len(CATEGORIES)
    }
    
# ========================
# Root
# ========================
@app.get("/")
def root():
    """API documentation."""
    return {
        "service": "Email Classification API",
        "version": "2.0.0",
        "endpoints": {
            "GET /health": "Health check",
            "GET /categories": "List available categories",
            "POST /classify": "Classify single email",
            "POST /batch-classify": "Classify multiple emails",
            "POST /feedback": "Record human feedback",
            "GET /stats": "Get classification statistics"
        },
        "model": "fine-tuned" if USE_FINETUNED else "zero-shot",
        "device": device
    }
    
@app.on_event("startup")
async def startup_event():
    """Log startup info."""
    logger.info("=" * 60)
    logger.info("🚀 Email Classification API Starting")
    logger.info(f"Device: {device}")
    logger.info(f"Fine-tuned model: {'✓ Loaded' if USE_FINETUNED else '✗ Not available'}")
    logger.info(f"Zero-shot model: {'✓ Loaded' if zero_shot_classifier else '✗ Not available'}")
    logger.info(f"Categories: {len(CATEGORIES)}")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    logger.info("=" * 60)
    
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down...")
    if finetuned_model:
        del finetuned_model
    if zero_shot_classifier:  # FIX: Added missing colon
        del zero_shot_classifier
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)