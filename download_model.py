import mlflow
import mlflow.pyfunc
import os

# Set the tracking URI for MLflow server
mlflow.set_tracking_uri("https://7030-2a02-2f04-530c-c000-511d-87f7-111d-1a6f.ngrok-free.app")

# Set the model URI and the destination directory
model_uri = "models:/spam_detector_raw/1"  # Replace with your model's name and version
model_dir = "./models/spam_detector_raw"  # Specify the directory where you want to save the model

# Download the model
model = mlflow.pyfunc.load_model(model_uri)

# Define a PythonModel subclass to wrap the model
class MyModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Save the model to the specified directory
mlflow.pyfunc.save_model(path=model_dir, python_model=MyModel())# ༼ つ ◕_◕ ༽つ
