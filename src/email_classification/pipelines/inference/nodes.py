"""Inference pipeline nodes for email classification."""
import torch
from pathlib import Path
from typing import Dict, List
import json
import logging
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)

logger = logging.getLogger(__name__)


def load_trained_model(model_path: str):
    """
    Load trained model and tokenizer.
    
    Args:
        model_path: Path to saved model directory
        
    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"📂 Loading trained model from {model_path}...")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded successfully on {device}")
    
    return tokenizer, model


def load_label_mapping(model_path: str) -> Dict[int, str]:
    """
    Load label mapping from model directory.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary mapping label IDs to label names
    """
    label_mapping_path = Path(model_path) / "label_mapping.json"
    
    if label_mapping_path.exists():
        with open(label_mapping_path) as f:
            mapping = json.load(f)
            return {int(k): v for k, v in mapping['id2label'].items()}
    
    logger.warning("Label mapping not found, using generic labels")
    return {}


def classify_emails(
    emails: List[Dict[str, str]],
    tokenizer,
    model,
    label_mapping: Dict[int, str],
    max_length: int = 512
) -> List[Dict]:
    """
    Classify a batch of emails.
    
    Args:
        emails: List of email dicts with 'subject' and 'body'
        tokenizer: Tokenizer instance
        model: Model instance
        label_mapping: Dictionary mapping label IDs to names
        max_length: Maximum sequence length
        
    Returns:
        List of classification results
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    for email in emails:
        # Prepare text
        text = f"Subject: {email.get('subject', '')} Body: {email.get('body', '')[:500]}"
        
        # Tokenize
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        
        # Get top prediction
        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx].cpu())
        
        category = label_mapping.get(pred_idx, f"Category_{pred_idx}")
        
        results.append({
            "email_id": email.get('id'),
            "category": category,
            "confidence": round(confidence, 4),
            "all_scores": {
                label_mapping.get(i, f"Category_{i}"): round(float(prob.cpu()), 4)
                for i, prob in enumerate(probs)
            }
        })
    
    logger.info(f"✓ Classified {len(emails)} emails")
    
    return results


def save_predictions(predictions: List[Dict], output_path: str) -> str:
    """
    Save classification predictions to file.
    
    Args:
        predictions: List of prediction results
        output_path: Path to save results
        
    Returns:
        Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Predictions saved to {output_path}")
    
    return output_path
