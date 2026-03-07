import os
import time
import numpy as np
import pickle
import argparse
from tensorflow import keras
import sounddevice as sd
import warnings

warnings.filterwarnings('ignore')

from ser_project.artifacts import SERDataLoaderArtifacts
from ser_project.training.features import extract_audio_features, extract_opensmile_features, extract_wav2vec_features

def record_audio(duration=3.0, sr=22050):
    print(f"\n[Microphone] Recording for {duration} seconds...")
    # Record audio with 1 channel (mono)
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("[Microphone] Recording complete.")
    # Return as 1D array
    return recording.flatten()

def main():
    parser = argparse.ArgumentParser(description="Real-time Speech Emotion Recognition")
    parser.add_argument("--feature_type", type=str, default="librosa", choices=["librosa", "opensmile", "wav2vec"],
                        help="Feature type the model was trained on")
    parser.add_argument("--model_path", type=str, default="ser_project/artifacts/ser_model.keras")
    parser.add_argument("--scaler_path", type=str, default="ser_project/artifacts/scaler.pkl")
    parser.add_argument("--duration", type=float, default=3.0, help="Duration to record per utterance")
    args = parser.parse_args()

    # Determine required sampling rate
    sr = 16000 if args.feature_type == 'wav2vec' else SERDataLoaderArtifacts.ser_res_type
    
    # We don't have sr=res_type directly if res_type is 'kaiser_fast', so let's use standard default
    if sr == 'kaiser_fast':
        sr = 22050
        
    print("Loading model and scaler...")
    if not os.path.exists(args.model_path) or not os.path.exists(args.scaler_path):
        print(f"Error: Could not find model at {args.model_path} or scaler at {args.scaler_path}.")
        print("Please run `python ser_project/training/train.py` first to generate them.")
        return

    model = keras.models.load_model(args.model_path)
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Reverse emotion map to get label from index
    emotion_map = SERDataLoaderArtifacts.emotion_map
    reverse_map = {int(v): k for k, v in emotion_map.items()}
    emotion_labels = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgusted', '08': 'surprised'
    }

    print("\n" + "="*50)
    print("SER Real-time Predictor")
    print(f"Feature Type: {args.feature_type}")
    print(f"Sampling Rate: {sr} Hz")
    print("="*50)

    while True:
        try:
            command = input("\nPress [Enter] to start recording (or 'q' to quit): ")
            if command.lower() == 'q':
                break

            # 1. Record audio
            audio_data = record_audio(duration=args.duration, sr=sr)
            
            # Remove silent ends to center the speech (optional but helpful)
            import librosa
            audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
            
            if len(audio_data) < sr * 0.5:
                print("Audio too short or silent. Try again.")
                continue

            # 2. Extract features
            try:
                if args.feature_type == 'wav2vec':
                    features = extract_wav2vec_features(audio_data)
                elif args.feature_type == 'opensmile':
                    features = extract_opensmile_features(audio_data, sr)
                else:
                    features = extract_audio_features(audio_data, sr)
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
                
            # 3. Scale and prepare for prediction
            features = features.reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # If CNN, expand dims
            if args.feature_type == 'librosa':
                features_scaled = np.expand_dims(features_scaled, axis=2)

            # 4. Predict
            predictions = model.predict(features_scaled, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx]
            
            mapped_code = list(emotion_map.keys())[list(emotion_map.values()).index(pred_idx)]
            emotion_name = emotion_labels[mapped_code]

            print(f">>> Predicted Emotion : {emotion_name.upper()} ({confidence*100:.2f}%)")

        except KeyboardInterrupt:
            break

    print("\nExiting...")

if __name__ == "__main__":
    main()
