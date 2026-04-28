"""Training pipeline definition."""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    load_and_prepare_data,
    create_train_test_split,
    load_tokenizer_and_model,
    tokenize_datasets,
    train_model,
    evaluate_model,
    save_model_and_metadata,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the email classification training pipeline.
    
    This pipeline implements the following workflow:
    1. Load and prepare raw email data
    2. Split into train/test sets
    3. Load tokenizer and model
    4. Tokenize datasets
    5. Train model
    6. Evaluate model
    7. Save model and metadata
    
    Returns:
        Kedro Pipeline object
    """
    return pipeline(
        [
            node(
                func=load_and_prepare_data,
                inputs="emails_raw_csv",
                outputs=["prepared_emails_data", "label_encoder"],
                name="load_and_prepare_data",
                tags=["data_preparation"],
            ),
            node(
                func=create_train_test_split,
                inputs=["prepared_emails_data", "params:test_size"],
                outputs=["train_dataset", "test_dataset", "n_train"],
                name="create_train_test_split",
                tags=["data_preparation"],
            ),
            node(
                func=load_tokenizer_and_model,
                inputs=["label_encoder", "params:model_name"],
                outputs=["tokenizer", "model", "num_labels"],
                name="load_tokenizer_and_model",
                tags=["model_setup"],
            ),
            node(
                func=tokenize_datasets,
                inputs=["train_dataset", "test_dataset", "tokenizer", "params:max_length"],
                outputs=["train_tokenized", "test_tokenized"],
                name="tokenize_datasets",
                tags=["data_processing"],
            ),
            node(
                func=train_model,
                inputs=[
                    "model",
                    "tokenizer",
                    "train_tokenized",
                    "test_tokenized",
                    "params:num_epochs",
                    "params:batch_size",
                    "params:learning_rate",
                    "params:output_dir",
                    "n_train",
                ],
                outputs="trained_model_data",
                name="train_model",
                tags=["training"],
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "trained_model_data",
                    "test_tokenized",
                    "label_encoder",
                ],
                outputs="evaluation_metrics",
                name="evaluate_model",
                tags=["evaluation"],
            ),
            node(
                func=save_model_and_metadata,
                inputs=["trained_model_data", "tokenizer", "label_encoder", "params:output_dir"],
                outputs="model_save_info",
                name="save_model_and_metadata",
                tags=["model_saving"],
            ),
        ]
    )
