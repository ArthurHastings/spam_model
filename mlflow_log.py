import mlflow.pyfunc
import os

from mlflow_func import SpamModelWrapper

mlflow.set_tracking_uri("http://localhost:5000")

artifacts = {
    "model": "spam_model.keras",
    "tokenizer": "tokenizer_spam.pkl"
}

mlflow.pyfunc.log_model(
    artifact_path="wrapped_spam_model",
    python_model=SpamModelWrapper(),
    artifacts={"model": os.path.join("artifacts", "spam_model.keras").replace("\\", "/")},
    registered_model_name="spam_detector_raw"
)
