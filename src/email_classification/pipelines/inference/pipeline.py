"""Inference pipeline definition."""
from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (
    load_trained_model,
    load_label_mapping,
    classify_emails,
    save_predictions,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Create the email classification inference pipeline.
    
    This pipeline implements the following workflow:
    1. Load trained model
    2. Load label mapping
    3. Classify new emails
    4. Save predictions
    
    Returns:
        Kedro Pipeline object
    """
    return pipeline(
        [
            node(
                func=load_trained_model,
                inputs="params:model_path",
                outputs=["tokenizer", "model"],
                name="load_trained_model",
                tags=["model_loading"],
            ),
            node(
                func=load_label_mapping,
                inputs="params:model_path",
                outputs="label_mapping",
                name="load_label_mapping",
                tags=["model_loading"],
            ),
            node(
                func=classify_emails,
                inputs=["emails_to_classify", "tokenizer", "model", "label_mapping", "params:max_length"],
                outputs="classification_results",
                name="classify_emails",
                tags=["inference"],
            ),
            node(
                func=save_predictions,
                inputs=["classification_results", "params:predictions_output_path"],
                outputs="predictions_file",
                name="save_predictions",
                tags=["output"],
            ),
        ]
    )
