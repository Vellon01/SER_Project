import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ser_project.training.features import extract_audio_features
from ser_project.artifacts import SERDataLoaderArtifacts
class SERDataLoader:
    def __init__(self, dataset_path = SERDataLoaderArtifacts.data_path):
        self.dataset_path = dataset_path
        self.emotion_map = SERDataLoaderArtifacts.emotion_map
    def process_dataset(self):
        x, y = [], []
        # Finding all .wav files in Actor directories
        search_path = os.path.join(self.dataset_path, SERDataLoaderArtifacts.ser_path_extension)
        for file in glob.glob(search_path):
            emotion_code = os.path.basename(file).split('-')[2]
            # Load and extract
            data, sr = librosa.load(file, res_type=SERDataLoaderArtifacts.ser_res_type)
            features = extract_audio_features(data, sr)
            x.append(features)
            y.append(self.emotion_map[emotion_code])
        return np.array(x), np.array(y)

    def prepare_pipeline_data(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Fit scaler on training data ONLY
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_val)
        
        # Reshape for CNN: (samples, features, 1 channel)
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        
        return x_train, x_test, y_train, y_val, scaler
    