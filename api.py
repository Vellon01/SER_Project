import os
import io
import pickle
import numpy as np
import librosa
import tempfile
import warnings

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pydantic import BaseModel
from tensorflow import keras

warnings.filterwarnings('ignore')

from ser_project.artifacts import SERDataLoaderArtifacts
from ser_project.training.features import extract_audio_features

# Global variables to hold model and scaler
model = None
scaler = None

# Reverse emotion map to get label from index
emotion_labels = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and scaler on API startup
    global model, scaler
    model_path = "ser_project/artifacts/ser_model.keras"
    scaler_path = "ser_project/artifacts/scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print("Loading model and scaler...")
        # Load Keras CNN model
        model = keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully.")
    else:
        print(f"Warning: Could not find model at {model_path} or scaler at {scaler_path}.")
        print("Please run `python ser_project/training/train.py` first to generate them.")
    
    yield
    # Cleanup on API shutdown
    model = None
    scaler = None

app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Real-time WebSocket and REST API for predicting emotion from audio",
    version="1.1.0",
    lifespan=lifespan
)

# Ensure static and templates exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class PredictionResponse(BaseModel):
    emotion: str
    confidence: float

def process_audio_and_predict(audio_data: np.ndarray, sr: int) -> dict:
    """Core logic to process audio, extract features, and predict emotion."""
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded.")

    # 1. Normalize audio to match training data intensity
    # Critical fix: Without this, quiet mic audio usually defaults to "disgusted" or "sad".
    audio_data = librosa.util.normalize(audio_data)

    # 2. Trim silence (top_db=20 effectively removes ambient room noise)
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    
    if len(audio_data) < sr * 0.5:
        raise ValueError("Audio chunk too short or contains only silence after trimming. Try speaking closer.")

    # 3. Extract features (Default librosa CNN features)
    features = extract_audio_features(audio_data, sr)
        
    # 4. Scale and prepare for prediction
    features = features.reshape(1, -1)
    features_scaled = scaler.transform(features)
    
    # 5. CNN expects 3D input (batch, features, 1)
    features_scaled = np.expand_dims(features_scaled, axis=2)

    # 6. Predict
    predictions = model.predict(features_scaled, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][pred_idx])
    
    emotion_map = SERDataLoaderArtifacts.emotion_map
    mapped_code = list(emotion_map.keys())[list(emotion_map.values()).index(pred_idx)]
    emotion_name = emotion_labels.get(mapped_code, "unknown")

    return {"emotion": emotion_name, "confidence": confidence}

def decode_audio_bytes(audio_bytes: bytes) -> tuple:
    """Decodes raw audio bytes into a librosa-compatible numpy array."""
    sr = SERDataLoaderArtifacts.ser_res_type
    if sr == 'kaiser_fast':
        sr = 22050

    # Write bytes to a temp file because librosa relies on standard audio decoding libs
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        audio_data, orig_sr = librosa.load(temp_path, sr=sr, mono=True)
        return audio_data, orig_sr
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio chunk streaming.
    Expects chunks of binary audio data (e.g., 2-4 seconds).
    """
    await websocket.accept()
    print("[WebSocket] Client connected.")
    try:
        while True:
            # Await the raw binary chunk (WAV, webm, etc.) from the client
            audio_bytes = await websocket.receive_bytes()
            
            try:
                # Threadpool is required to avoid blocking the async event loop during decoding/inference
                audio_data, sr = await run_in_threadpool(decode_audio_bytes, audio_bytes)
                result = await run_in_threadpool(process_audio_and_predict, audio_data, sr)
                
                await websocket.send_json(result)
            except ValueError as ve:
                await websocket.send_json({"error": str(ve)})
            except RuntimeError as re:
                await websocket.send_json({"error": str(re)})
            except Exception as e:
                print(f"[WebSocket] Error during prediction: {e}")
                await websocket.send_json({"error": "Internal processing error."})
                
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion_rest(file: UploadFile = File(...)):
    """
    Standard REST endpoint for file upload testing.
    """
    try:
        audio_bytes = await file.read()
        audio_data, sr = await run_in_threadpool(decode_audio_bytes, audio_bytes)
        result = await run_in_threadpool(process_audio_and_predict, audio_data, sr)
        
        return PredictionResponse(**result)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
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
