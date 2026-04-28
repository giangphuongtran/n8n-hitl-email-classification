"""Project pipelines."""
from __future__ import annotations

from kedro.pipeline import Pipeline
from email_classification.pipelines.training import create_pipeline as create_training_pipeline
from email_classification.pipelines.inference import create_pipeline as create_inference_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    training_pipeline = create_training_pipeline()
    inference_pipeline = create_inference_pipeline()
    
    return {
        "training": training_pipeline,
        "inference": inference_pipeline,
        "__default__": inference_pipeline,
    }
