import mlflow.pyfunc
import os
import pickle
import tensorflow as tf
from mlflow_func import SpamModelWrapper

# No need to set tracking URI here if using ngrok directly
mlflow.set_tracking_uri("https://7030-2a02-2f04-530c-c000-511d-87f7-111d-1a6f.ngrok-free.app")
# Assuming MLflow server is running with ngrok on the GitHub Actions environment

artifacts = {
    "model": "spam_model/spam_model.keras",
    "tokenizer": "spam_model/tokenizer_spam.pkl"
}

mlflow.pyfunc.log_model(
    artifact_path="wrapped_spam_model",
    python_model=SpamModelWrapper(),
    artifacts=artifacts,
    registered_model_name="spam_detector_raw"
)