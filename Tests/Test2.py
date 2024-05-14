import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyaudio
import wave
import librosa

# Load the pre-trained model
model = load_model('Best-RNN-LSTM-MODEL.h5')

# Function to record audio
def record_audio(duration=3, fs=16000):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    print("Recording...")
    frames = []

    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Done recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    waveFile = wave.open('recorded.wav', 'wb')
    waveFile.setnchannels(1)
    waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(fs)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    return 'recorded.wav'

# Function to preprocess audio
def preprocess_audio(file_path, target_sr=16000):
    y, sr = librosa.load(file_path, sr=target_sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=0)

# Function to predict using the model
def predict(file_path):
    processed_audio = preprocess_audio(file_path)
    prediction = model.predict(processed_audio)
    return prediction

# Main script
if __name__ == "__main__":
    audio_file = record_audio()
    prediction = predict(audio_file)
    print(f"Prediction: {prediction}")

    
