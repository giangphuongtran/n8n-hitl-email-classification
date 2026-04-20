"""Email Classification Pipelines."""
from .training import create_pipeline as create_training_pipeline
from .inference import create_pipeline as create_inference_pipeline

__all__ = ["create_training_pipeline", "create_inference_pipeline"]
