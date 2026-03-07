import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ser_project.training.features import extract_audio_features, extract_opensmile_features, extract_wav2vec_features
from ser_project.artifacts import SERDataLoaderArtifacts
class SERDataLoader:
    def __init__(self, dataset_path=SERDataLoaderArtifacts.data_path, feature_type='librosa'):
        self.dataset_path = dataset_path
        self.emotion_map = SERDataLoaderArtifacts.emotion_map
        self.feature_type = feature_type
    def process_dataset(self, augment=True):
        x, y = [], []
        # Finding all .wav files in Actor directories
        search_path = os.path.join(self.dataset_path, SERDataLoaderArtifacts.ser_path_extension)
        
        def _extract_and_store(audio_data, sr, em_code, is_w2v):
            if is_w2v:
                features = extract_wav2vec_features(audio_data)
            else:
                if self.feature_type == 'opensmile':
                    features = extract_opensmile_features(audio_data, sr)
                else:
                    features = extract_audio_features(audio_data, sr)
            x.append(features)
            y.append(self.emotion_map[em_code])

        for file in glob.glob(search_path):
            emotion_code = os.path.basename(file).split('-')[2]
            
            # Load and extract based on feature_type
            if self.feature_type == 'wav2vec':
                data, sr = librosa.load(file, sr=16000, res_type=SERDataLoaderArtifacts.ser_res_type)
                _extract_and_store(data, sr, emotion_code, is_w2v=True)
                
                if augment:
                    # Add subtle noise variation for wav2vec
                    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
                    noisy_data = data + noise_amp * np.random.normal(size=data.shape[0])
                    _extract_and_store(noisy_data, sr, emotion_code, is_w2v=True)
            else:
                data, sr = librosa.load(file, res_type=SERDataLoaderArtifacts.ser_res_type)
                _extract_and_store(data, sr, emotion_code, is_w2v=False)
                
                if augment:
                    # 1. Noise Injection
                    noise_amp = 0.005 * np.random.uniform() * np.amax(data)
                    noisy_data = data + noise_amp * np.random.normal(size=data.shape[0])
                    _extract_and_store(noisy_data, sr, emotion_code, is_w2v=False)
                    
                    # 2. Pitch Shift (only for non-wav2vec to save processing time)
                    pitched_data = librosa.effects.pitch_shift(y=data, sr=sr, n_steps=0.7)
                    _extract_and_store(pitched_data, sr, emotion_code, is_w2v=False)
            
        return np.array(x), np.array(y)

    def prepare_pipeline_data(self, x, y, expand_dims_for_cnn=True):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Fit scaler on training data ONLY
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_val)
        
        if expand_dims_for_cnn:
            # Reshape for CNN: (samples, features, 1 channel)
            x_train = np.expand_dims(x_train, axis=2)
            x_test = np.expand_dims(x_test, axis=2)
        
        return x_train, x_test, y_train, y_val, scaler
    