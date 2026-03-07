import numpy as np
import librosa

def extract_features_librosa(data, sr):
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    
    # Chroma
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    
    # Mel Frequency Cepstral Coefficients
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    
    # Stack them into one long array
    # Shape: (1 + 12 + 40) = 53 features
    result = np.hstack((zcr, chroma, mfccs))
    return result

import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)
def extract_features_opensmile(data, sr):
    # Process the audio array directly
    features_df = smile.process_signal(data, sr)
    # Extract the first row as a 1D NumPy array (88 features)
    return features_df.iloc[0].to_numpy()

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
# --- Wav2Vec 2.0 Init ---
print("Loading Wav2Vec 2.0 model...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model_w2v = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract_features_wav2vec(data_16k):
    # Process audio (Wav2Vec strictly expects 16kHz audio)
    inputs = processor(data_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # Extract deep embeddings without training the base model
    with torch.no_grad():
        outputs = model_w2v(**inputs)
    
    # Take the mean across the time dimension to get a 1D array of length 768
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

import os
import glob
from pathlib import Path

emotion_map = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'
}

def load_ravdess_data(dataset_path, feature_type='librosa'):
    data = []
    labels = []
    emotions = []
    
    actor_dirs = sorted(glob.glob(os.path.join(dataset_path, 'Actor_*')))
    
    for actor_dir in actor_dirs:
        wav_files = sorted(glob.glob(os.path.join(actor_dir, '*.wav')))
        
        for wav_file in wav_files:
            filename = os.path.basename(wav_file).split('.')[0]
            emotion_code = filename.split('-')[2]
            emotion_label = emotion_map.get(emotion_code, 'unknown')
            
            # Route to the correct extractor and sampling rate
            if feature_type == 'wav2vec':
                y, sr = librosa.load(wav_file, sr=16000) # STRICT 16kHz requirement
                features = extract_features_wav2vec(y)
            else:
                y, sr = librosa.load(wav_file) # Default sampling rate
                if feature_type == 'opensmile':
                    features = extract_features_opensmile(y, sr)
                else:
                    features = extract_features_librosa(y, sr)
            
            data.append(features)
            emotions.append(emotion_label)
            labels.append(int(emotion_code) - 1) 
    
    return np.array(data), np.array(labels), emotions

dataset_path = r'c:\Users\vello\OneDrive\Desktop\imp\research\SER_Project\datasets\uwrfkaggler\ravdess-emotional-speech-audio\versions\1\audio_speech_actors_01-24'

# --- THE COMPARISON TOGGLE ---
# Options: 'librosa', 'opensmile', 'wav2vec'
CHOSEN_EXTRACTOR = 'opensmile' 

print(f"Loading data using {CHOSEN_EXTRACTOR}...")
X, y, emotions = load_ravdess_data(dataset_path, feature_type=CHOSEN_EXTRACTOR)

print(f"Data shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Unique emotions: {set(emotions)}")
print(f"Emotion distribution: {np.bincount(y)}")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Reshape data for CNN (add channel dimension)
X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Test set shape: {X_test_scaled.shape}")
print(f"Number of classes: {len(set(y))}")

# Build CNN model
model = keras.Sequential([
    layers.Conv1D(64, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    
    layers.Conv1D(128, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    
    layers.Conv1D(256, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(8, activation='softmax')  # 8 emotions
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Predictions
y_pred = model.predict(X_test_scaled).argmax(axis=1)

# Classification report
print("\nClassification Report:")
emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
print(classification_report(y_test, y_pred, target_names=emotion_labels))

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()
ax1.grid()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()
ax2.grid()

plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# import pickle
# # SAVE
# model.save('ser_model.h5')
# pickle.dump(scaler, open('scaler.pkl', 'wb'))

# # LOAD (in a new notebook/script)
# model = keras.models.load_model('ser_model.h5')
# scaler = pickle.load(open('scaler.pkl', 'rb'))




