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