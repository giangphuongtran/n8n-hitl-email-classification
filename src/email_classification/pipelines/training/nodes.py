"""Training pipeline nodes for email classification."""
import pandas as pd
import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


def load_and_prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Load CSV and prepare texts + labels.
    
    Args:
        csv_path: Path to CSV file with emails and labels
        
    Returns:
        Tuple of (processed dataframe, label encoder)
    """
    logger.info(f"📂 Loading data from Catalog...")
    
    # Remove rows with missing labels
    df = df.dropna(subset=['CorrectedLabel'])
    logger.info(f"✓ Loaded {len(df)} emails with labels")
    
    # Label Encoding
    label_encoder = LabelEncoder()
    df['labels'] = label_encoder.fit_transform(df['CorrectedLabel'])
    
    # Combine subject and body
    body_col = 'Body' if 'Body' in df.columns else 'Snippet'
    df['text'] = (
        "Subject: " + df['Subject'].fillna('').str.strip() +
        " Body: " + df[body_col].fillna('').str.strip()
    )
    
    # Remove very short texts (less than 10 chars)
    df = df[df['text'].str.len() > 10]
    logger.info(f"✓ After filtering short emails: {len(df)} emails remain")
    
    # Show class distribution
    logger.info("📊 Class distribution:")
    for label, count in df['CorrectedLabel'].value_counts().items():
        pct = 100 * count / len(df)
        logger.info(f"  {label}: {count} ({pct:.1f}%)")
    
    return df, label_encoder


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[Dataset, Dataset, int]:
    """
    Create train/test split with stratification.
    
    Args:
        df: Input dataframe with text and labels
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (train_dataset, test_dataset, n_train)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['CorrectedLabel'],
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"📐 Dataset split:")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")
    
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


def load_tokenizer_and_model(
    label_encoder: LabelEncoder,
    model_name: str = "distilbert-base-multilingual-cased"
) -> Tuple:
    """
    Load tokenizer and model.
    
    Args:
        label_encoder: Fitted label encoder with classes
        model_name: Hugging Face model name
        
    Returns:
        Tuple of (tokenizer, model, num_labels)
    """
    logger.info(f"🤗 Loading {model_name} tokenizer and model...")
    
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    num_labels = len(label_encoder.classes_)
    
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )
    
    logger.info(f"✓ Model loaded with {num_labels} labels")
    logger.info(f"  Labels: {list(label_encoder.classes_)}")
    
    return tokenizer, model, num_labels


def tokenize_datasets(
    train_dataset: Dataset,
    test_dataset: Dataset,
    tokenizer,
    max_length: int = 512
) -> Tuple[Dataset, Dataset]:
    """
    Tokenize train and test datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (tokenized_train, tokenized_test)
    """
    logger.info(f"⚙️  Tokenizing datasets (max_length={max_length})...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    train_tokenized = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    test_tokenized = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['text']
    )
    
    logger.info("✓ Tokenization complete")
    return train_tokenized, test_tokenized


def train_model(
    model,
    tokenizer,
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "./email_classifier_finetuned",
    n_train: int = 1000,
) -> Dict[str, Any]:
    """
    Train the model using Hugging Face Trainer.
    
    Args:
        model: DistilBERT model instance
        tokenizer: Tokenizer instance
        train_dataset: Tokenized training dataset
        test_dataset: Tokenized test dataset
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        output_dir: Directory to save model
        n_train: Number of training samples
        
    Returns:
        Dictionary with training metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🖥️  Using device: {device}")
    
    logger.info(f"🎯 Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    
    warmup_steps = max(100, int(n_train / batch_size * 0.1))
    
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
        fp16=device == "cuda",
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )
    
    logger.info("🚀 Starting training...")
    trainer.train()
    
    return {"trainer": trainer, "training_args": training_args}


def evaluate_model(
    trainer,
    test_dataset: Dataset,
    label_encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    Evaluate the trained model.
    
    Args:
        trainer: Hugging Face Trainer instance
        test_dataset: Tokenized test dataset
        label_encoder: Fitted label encoder
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("📈 Evaluating model on test set...")
    
    eval_results = trainer.evaluate()
    logger.info(f"✓ Test Accuracy: {eval_results['eval_accuracy']:.4f}")
    
    # Detailed evaluation
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    test_labels = test_dataset['labels']
    
    logger.info(f"📊 Detailed Classification Report:")
    report = classification_report(
        test_labels,
        preds,
        target_names=label_encoder.classes_,
        digits=4,
        output_dict=True
    )
    
    cm = confusion_matrix(test_labels, preds)
    
    return {
        "accuracy": eval_results['eval_accuracy'],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "test_labels": test_labels
    }


def save_model_and_metadata(
    trainer,
    tokenizer,
    label_encoder: LabelEncoder,
    output_dir: str = "./email_classifier_finetuned"
) -> Dict[str, str]:
    """
    Save trained model and metadata.
    
    Args:
        trainer: Hugging Face Trainer instance
        tokenizer: Tokenizer instance
        label_encoder: Fitted label encoder
        output_dir: Directory to save model
        
    Returns:
        Dictionary with save paths
    """
    logger.info(f"💾 Saving model to {output_dir}...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label encoder metadata
    label_mapping = {
        'label2id': {label: i for i, label in enumerate(label_encoder.classes_)},
        'id2label': {i: label for i, label in enumerate(label_encoder.classes_)}
    }
    
    label_mapping_path = f"{output_dir}/label_mapping.json"
    with open(label_mapping_path, 'w') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Model saved to {output_dir} with label mapping")
    
    return {
        "model_path": output_dir,
        "label_mapping_path": label_mapping_path
    }
