import os
import pickle
import numpy as np
import librosa
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from tensorflow import keras

from ser_project.artifacts import SERDataLoaderArtifacts
from ser_project.training.features import extract_audio_features, extract_opensmile_features, extract_wav2vec_features

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for predicting emotion from audio files",
    version="1.0.0"
)

# Global variables to hold model and scaler
model = None
scaler = None

# Reverse emotion map to get label from index
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'
}

@app.on_event("startup")
async def startup_event():
    global model, scaler
    model_path = "ser_project/artifacts/ser_model.keras"
    scaler_path = "ser_project/artifacts/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading model and scaler...")
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully.")
    else:
        print(f"Warning: Could not find model at {model_path} or scaler at {scaler_path}.")

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(
    file: UploadFile = File(...),
    feature_type: str = Form("librosa", description="Feature type to use: librosa, opensmile, or wav2vec")
):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    if feature_type not in ["librosa", "opensmile", "wav2vec"]:
        raise HTTPException(status_code=400, detail="Invalid feature_type. Choose from: librosa, opensmile, wav2vec")

    try:
        audio_bytes = await file.read()
        
        # Save to a temporary file so librosa can load it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            temp_path = temp_audio.name

        sr = 16000 if feature_type == 'wav2vec' else SERDataLoaderArtifacts.ser_res_type
        if sr == 'kaiser_fast':
            sr = 22050

        audio_data, _ = librosa.load(temp_path, sr=sr, mono=True)
        os.remove(temp_path)

        # Remove silent ends
        audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
        
        if len(audio_data) < sr * 0.5:
            raise HTTPException(status_code=400, detail="Audio file too short or contains only silence.")

        # Extract features
        if feature_type == 'wav2vec':
            features = extract_wav2vec_features(audio_data)
        elif feature_type == 'opensmile':
            features = extract_opensmile_features(audio_data, sr)
        else:
            features = extract_audio_features(audio_data, sr)
            
        # Scale and prepare for prediction
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        # If CNN, expand dims
        if feature_type == 'librosa':
            features_scaled = np.expand_dims(features_scaled, axis=2)

        # Predict
        predictions = model.predict(features_scaled, verbose=0)
        pred_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_idx])
        
        emotion_map = SERDataLoaderArtifacts.emotion_map
        mapped_code = list(emotion_map.keys())[list(emotion_map.values()).index(pred_idx)]
        emotion_name = emotion_labels.get(mapped_code, "unknown")

        return PredictionResponse(emotion=emotion_name, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.get("/health")
def health_check():
    """Endpoint for checking API health and model status"""
    return {
        "status": "ok", 
        "model_loaded": model is not None, 
        "scaler_loaded": scaler is not None
    }
