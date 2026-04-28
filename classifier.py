"""
Fine-tune DistilBERT for email classification with human feedback loop.
Optimized for 50-3000 multilingual email samples (Vietnamese + English).
 
Usage:
    python classifier.py --data training_data.csv --output email_classifier_finetuned --epochs 3 --batch-size 16
"""

import sys

print("Starting classifier imports...", flush=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import argparse
import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

def load_and_prepare_data(csv_path: str) -> tuple:
    """Load CSV and prepare texts + labels."""
    print(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing labels
    df = df.dropna(subset=['CorrectedLabel'])
    print(f"✓ Loaded {len(df)} emails with labels")
    
    # Label Encoding - FIX: Use same encoder instance
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['CorrectedLabel'])
    
    # Combine subject and body - FIX: Handle both 'Body' and 'Snippet' columns
    body_col = 'Body' if 'Body' in df.columns else 'Snippet'
    df['text'] = (
        "Subject: " + df['Subject'].fillna('').str.strip() +
        " Body: " + df[body_col].fillna('').str.strip()
    )
    
    # Remove very short texts (less than 10 chars)
    df = df[df['text'].str.len() > 10]
    print(f"✓ After filtering short emails: {len(df)} emails remain")
    
    # Show class distribution
    print(f"\n📊 Class distribution:")
    for label, count in df['CorrectedLabel'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    return df, label_encoder

def create_dataset(df: pd.DataFrame, test_size: float = 0.2):
    """Create train/test split with stratification."""
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['CorrectedLabel'],
        random_state=42,
        shuffle=True
    )
    
    print(f"\n📐 Dataset split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Create Dataset objects
    train_dataset = Dataset.from_dict({
        'text': train_df['text'].tolist(),
        'labels': train_df['labels'].tolist()
    })
    
    test_dataset = Dataset.from_dict({
        'text': test_df['text'].tolist(),
        'labels': test_df['labels'].tolist()
    })
    
    return train_dataset, test_dataset, len(train_df)

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize texts"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )
    
def train_classifier(
    csv_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    warmup_ratio: float = 0.1,
):
    """Main training pipeline."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    
    # Step 1: Load and prepare data
    df, label_encoder = load_and_prepare_data(csv_path)
    num_labels = len(label_encoder.classes_)
    
    # Step 2: Create datasets
    train_dataset, test_dataset, n_train = create_dataset(df, test_size=0.2)
    print(f"✓ Number of training samples: {n_train}")
    
    # Step 3: Load tokenizer and model
    print(f"\n🤗 Loading DistilBERT tokenizer and model...")
    model_name = "distilbert-base-multilingual-cased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    
    print(f"✓ Model loaded with {num_labels} labels")
    print(f"  Labels: {list(label_encoder.classes_)}")
    
    # Step 4: Tokenize datasets
    print(f"\n⚙️  Tokenizing datasets...")
    tokenize_fn = lambda x: tokenize_function(x, tokenizer, max_length)
    
    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    test_tokenized = test_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])
    print("✓ Tokenization complete")
    
    # Step 5: Set up trainer with appropriate hyperparameters
    print(f"\n🎯 Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")
    print(f"  Warmup ratio: {warmup_ratio}")
    
    # Adjust parameters based on dataset size
    warmup_steps = max(100, int(n_train / batch_size * warmup_ratio))
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=1,
        seed=42,
        optim="adamw_torch",
        fp16=device == "cuda",  # Use mixed precision on GPU
    )
    
    # Custom metrics
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)   
    )
    
    # Step 6: Train the model
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Step 7: Evaluate the model
    print("\n📈 Evaluating model on test set...")
    eval_results = trainer.evaluate()
    print(f"✓ Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Detailed evaluation
    predictions = trainer.predict(test_tokenized)
    preds = np.argmax(predictions.predictions, axis=1)
    test_labels = test_tokenized['labels']
    
    print(f"\n📊 Detailed Classification Report:")
    print(classification_report(
        test_labels,
        preds,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, preds)
    print("📋 Confusion Matrix:")
    print(cm)
    
    # Step 8: Save the model and label encoder
    print(f"\n💾 Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label encoder
    label_mapping = {
        'label2id': {label: i for i, label in enumerate(label_encoder.classes_)},
        'id2label': {i: label for i, label in enumerate(label_encoder.classes_)}
    }
    
    with open(f"{output_dir}/label_mapping.json", 'w') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Model saved to {output_dir} with label mapping")
    
    # Step 9: Summary
    print("\n" + "="*60)
    print("🎉 Training complete!")
    print("="*60)
    print(f"Model: {model_name} (multilingual)")
    print(f"Classes: {num_labels}")
    print(f"Training samples: {n_train}")
    print(f"Test Accuracy: {eval_results['eval_accuracy']:.2%}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    return model, tokenizer, label_encoder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for email classification.")
    parser.add_argument(
        "--data",
        type=str,
        default="training_data.csv",
        help="Path to CSV file with Subject, Snippet/Body, and CorrectedLabel columns"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="email_classifier_finetuned",
        help="Output directory for model and tokenizer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (3 recommended for 3000+ samples)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (16-32 for 3000+ samples)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    print(
        f"Arguments parsed. Data={args.data}, output={args.output}, "
        f"epochs={args.epochs}, batch_size={args.batch_size}, max_length={args.max_length}",
        flush=True,
    )
    
    train_classifier(
        csv_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )
