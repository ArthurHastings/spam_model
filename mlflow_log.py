import mlflow.pyfunc

from mlflow_func import SpamModelWrapper

mlflow.set_tracking_uri("https://98f7-2a02-2f04-530c-c000-511d-87f7-111d-1a6f.ngrok-free.app")

artifacts = {
    "model": "./spam_model.keras",
    "tokenizer": "./tokenizer_spam.pkl"
}

mlflow.pyfunc.log_model(
    artifact_path="wrapped_spam_model",
    python_model=SpamModelWrapper(),
    artifacts=artifacts,
    registered_model_name="spam_detector_raw"
)