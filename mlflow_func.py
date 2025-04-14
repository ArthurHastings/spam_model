import mlflow.pyfunc
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

class SpamModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = tf.keras.models.load_model(context.artifacts["model"])
        with open(context.artifacts["tokenizer"], "rb") as f:
            self.tokenizer = pickle.load(f)
        self.max_len = 100

    def predict(self, context, model_input):
        texts = model_input["text"].tolist()
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len)
        return self.model.predict(padded)


# mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///C:/Users/User/Desktop/PythonAICourse/mlflow_artifacts --host 0.0.0.0 --port 5000