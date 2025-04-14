import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dropout
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import mlflow
import itertools
import time
import pickle

# 1️⃣ DESCĂRCARE AUTOMATĂ A SETULUI DE DATE
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
dataset_path = "smsspamcollection.zip"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SPAM_DETECTION_MODELV1")

if not os.path.exists(dataset_path):
    print("🔽 Descărcare dataset...")
    urllib.request.urlretrieve(url, dataset_path)
    print("✅ Descărcare finalizată!")

# Dezarhivare
with zipfile.ZipFile(dataset_path, "r") as zip_ref:
    zip_ref.extractall(".")

# Citire date
data_path = "SMSSpamCollection"
df = pd.read_csv(data_path, sep="\t", names=["label", "text"], header=None)

# Convertim etichetele (spam = 1, ham = 0)
df["label"] = df["label"].map({"spam": 1, "ham": 0})


# 2️⃣ PREPROCESARE TEXT
max_words = 5000
max_len = 100

# Tokenizare
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(df["text"])

# Convertire text -> secvențe numerice
X = tokenizer.texts_to_sequences(df["text"])
X = pad_sequences(X, maxlen=max_len)

# Etichete
y = df["label"].values

# 3️⃣ ÎMPĂRȚIREA ÎN TRAIN ȘI TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


optimizers = ["adam"]
batch_sizes = [32]
epochs_list = [35]
neurons = [64]
dropouts = [0.3]
learning_rates = [0.0005]

print("ZAZA")

param_combinations = list(itertools.product(optimizers, batch_sizes, epochs_list, neurons, dropouts, learning_rates))

for run_id, (optimizer, batch_size, epochs, neuron, dropout, learning_rate) in enumerate(param_combinations):
    with mlflow.start_run():
        
        run_name = f"Run_{run_id+1}_Opt-{optimizer}_BS-{batch_size}_Ep-{epochs}_Neurons-{neuron}_Dropout-{dropout}_LR-{learning_rate}"

        mlflow.set_tag("mlflow.runName", run_name)
        print(f"Starting {run_name}")

        model = keras.models.Sequential([
            keras.layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
            
            keras.layers.LSTM(neuron, return_sequences=True),  
            keras.layers.LSTM(neuron//2),

            keras.layers.Dropout(dropout),
            
            keras.layers.Dense(neuron//4, activation='leaky_relu'),
            keras.layers.Dropout(dropout),

            keras.layers.Dense(1, activation='sigmoid')
        ])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)

        if optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate = learning_rate)
        elif optimizer == "adamw":
            optimizer = tf._optimizers.AdamW(learning_rate = learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("neurons", neuron)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("learning rate", learning_rate)

        start_time = time.time()

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])

        training_time = time.time() - start_time
        mlflow.log_metric("training_time", training_time)

        loss, accuracy = model.evaluate(X_test, y_test)

        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        # print(f"Test Accuracy: {test_accuracy}")

        for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

        mlflow.tensorflow.log_model(model, f"cnn_model_run{run_id+1}")

        mlflow.end_run()

model.save("spam_model.keras")

with open("tokenizer_spam.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# -----------------------------------------------------------------------------------------------------------------------------------------------


# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs_range = range(len(acc))

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')

# plt.show()

# # -----------------------------------------------------------------------------------------------------------------------------------------------


# sample_texts = ["Congratulations! You won a free gift. Click here to claim!",
#                 "Hey, are we still meeting for lunch?",
#                 "WIN a lottery of $1000 NOW!",
#                 "Can you call me later?"]
# sample_seq = tokenizer.texts_to_sequences(sample_texts)
# sample_seq = pad_sequences(sample_seq, maxlen=max_len)

# predictions = model.predict(sample_seq)

# print("\n📌 Exemple de predicții:")
# for text, pred in zip(sample_texts, predictions):
#     print(f"Mesaj: {text} -> Spam Probability: {pred[0]:.2%}")