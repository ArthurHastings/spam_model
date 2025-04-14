import tensorflow as tf
import mlflow
import pickle

class SpamModelWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts["model"]
        tokenizer_path = context.artifacts["tokenizer"]

        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

    def predict(self, context, model_input):
        sequences = self.tokenizer.texts_to_sequences(model_input["text"])
        # padding etc. here if needed
        return self.model.predict(sequences)