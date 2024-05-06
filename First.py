import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers

# Load data from CSV
data = pd.read_csv("speech_data.csv")

# Data preprocessing
def preprocess_data(data):
    # Load audio files and extract features (e.g., MFCCs)
    features = []
    for audio_file_path in data['Audio File Path']:
        audio, sr = librosa.load(audio_file_path, sr=None)  # Load audio file
        mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=13)  # Extract MFCCs
        features.append(mfccs)
    features = np.array(features)
    
    # Tokenize transcriptions (convert text to numerical tokens)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['Transcription'])
    transcriptions = tokenizer.texts_to_sequences(data['Transcription'])
    transcriptions = tf.keras.preprocessing.sequence.pad_sequences(transcriptions)  # Pad sequences to ensure uniform length
    
    return features, transcriptions, tokenizer

# Split data into train, validation, and test sets
def split_data(features, transcriptions):
    X_train, X_val_test, y_train, y_val_test = train_test_split(features, transcriptions, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Model architecture
def build_model(input_shape, output_vocab_size):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.Dense(output_vocab_size, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and preprocess data
features, transcriptions, tokenizer = preprocess_data(data)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(features, transcriptions)

# Build model
model = build_model(input_shape=X_train[0].shape, output_vocab_size=len(tokenizer.word_index)+1)

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Save model
model.save("speech_recognition_model.h5")
# then i will convert it to tflite
