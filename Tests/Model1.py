import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Data parameters
data_dir = "path/to/your/data/directory"  # Replace with the actual path
n_commands = 8
n_mfccs = 13
sequence_length = 100  # Adjust based on your data

# Load and preprocess data
def load_data(data_dir):
    audio_data = []
    labels = []
    for command_dir in os.listdir(data_dir):
        command_path = os.path.join(data_dir, command_dir)
        for filename in os.listdir(command_path):
            audio_path = os.path.join(command_path, filename)
            audio, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfccs)
            mfccs = mfccs.T  # Transpose for LSTM input
            audio_data.append(mfccs)
            labels.append(int(command_dir))  # Assuming command directories are named with integers

    audio_data = np.array(audio_data)
    labels = np.array(labels)
    return audio_data, labels

# Load data
audio_data, labels = load_data(data_dir)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(audio_data, labels, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)

# Define model
inputs = Input(shape=(sequence_length, n_mfccs))
x = LSTM(128, return_sequences=True)(inputs)
x = LSTM(64)(x)
outputs = Dense(n_commands, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate model
loss, accuracy = model.evaluate(X_val, y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
