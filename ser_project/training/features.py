import numpy as np
import librosa

def extract_audio_features(data, sr):
    """Extracts ZCR, Chroma, and MFCC features from raw audio."""
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    
    # Chroma
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    
    # MFCCs (Mel Frequency Cepstral Coefficients)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    
    # Stack into a single feature vector
    return np.hstack((zcr, chroma, mfccs))

# --- OpenSMILE ---
_SMILE_INSTANCE = None

def get_smile_instance():
    global _SMILE_INSTANCE
    if _SMILE_INSTANCE is None:
        import opensmile
        _SMILE_INSTANCE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals
        )
    return _SMILE_INSTANCE

def extract_opensmile_features(data, sr):
    """Extracts ComParE_2016 features using OpenSMILE."""
    smile = get_smile_instance()
    # Process the audio array directly
    features_df = smile.process_signal(data, sr)
    # Extract the first row as a 1D NumPy array (88 features)
    return features_df.iloc[0].to_numpy()

# --- Wav2Vec 2.0 ---
_WAV2VEC_PROCESSOR = None
_WAV2VEC_MODEL = None

def get_wav2vec_model():
    global _WAV2VEC_PROCESSOR, _WAV2VEC_MODEL
    if _WAV2VEC_MODEL is None:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        # Load Wav2Vec 2.0 model
        _WAV2VEC_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _WAV2VEC_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return _WAV2VEC_PROCESSOR, _WAV2VEC_MODEL

def extract_wav2vec_features(data_16k):
    """Extracts deep embeddings from Wav2Vec 2.0 (Strictly expects 16kHz audio)."""
    import torch
    processor, model_w2v = get_wav2vec_model()
    
    inputs = processor(data_16k, sampling_rate=16000, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model_w2v(**inputs)
    
    # Take the mean across the time dimension to get a 1D array of length 768
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings
