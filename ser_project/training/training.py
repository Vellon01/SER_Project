from tensorflow.keras import layers, models

def build_ser_cnn(input_shape):
    """Defines a 1D-CNN architecture for SER (Restored to original high-accuracy depth)."""
    model = models.Sequential([
        # Use explicit Input layer to avoid Keras 3 warnings
        layers.Input(shape=input_shape),
        
        layers.Conv1D(64, 3, activation='relu'),
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
        
        layers.Dense(8, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_ser_dense(input_shape):
    """Defines a fully connected Dense architecture for 1D embeddings (e.g. wav2vec, opensmile)."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(8, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model