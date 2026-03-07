from ser_project.logger import Logging  # Use the full path from src
import pickle
from ser_project.training.data_ingestion import SERDataLoader
from ser_project.training.training import build_ser_cnn, build_ser_dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

logger = Logging()
class SERTrainingPipeline:
    def __init__(self, feature_type='librosa'):
        self.feature_type = feature_type

    def run_training_pipeline(self, raw_data_path):
        try:
            logger.info(f"--- Phase 1: Ingesting Raw Audio with {self.feature_type} ---")
            loader = SERDataLoader(raw_data_path, feature_type=self.feature_type)
            x_raw, y_raw = loader.process_dataset()

            logger.info("--- Phase 2: Preprocessing and Reshaping ---")
            is_cnn = self.feature_type == 'librosa'
            x_train, x_test, y_train, y_test, scaler = loader.prepare_pipeline_data(x_raw, y_raw, expand_dims_for_cnn=is_cnn)

            logger.info(f"--- Phase 3: Training Model ({'CNN' if is_cnn else 'Dense'}) ---")
            if is_cnn:
                model = build_ser_cnn(input_shape=(x_train.shape[1], 1))
            else:
                model = build_ser_dense(input_shape=(x_train.shape[1],))
            
            # Setup Callbacks
            os.makedirs('ser_project/artifacts', exist_ok=True)
            model_path = 'ser_project/artifacts/ser_model.keras'
            
            callbacks = [
                ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_accuracy', mode='max'),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1),
                EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
            ]
                
            history = model.fit(
                x_train, y_train, 
                epochs=50, 
                batch_size=32, 
                validation_split=0.1,
                callbacks=callbacks
            )

            logger.info("--- Phase 4: Evaluation ---")
            loss, acc = model.evaluate(x_test, y_test)
            logger.info(f"Pipeline Test Accuracy: {acc:.4f}")

            logger.info("--- Phase 5: Saving Pipeline Artifacts ---")
            try:
                # Note: The best model weights are already saved by ModelCheckpoint during training
                # and restored by EarlyStopping. We just save the scaler here.
                with open('ser_project/artifacts/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info("Pipeline execution complete. Best model saved to artifacts/ser_model.keras")
                return x_test, y_test
            except Exception as save_err:
                logger.error(f"Error saving artifacts: {save_err}")
                return None, None

        except FileNotFoundError as fnf_err:
            logger.error(f"File not found: {fnf_err}")
        except ValueError as val_err:
            logger.warning(f"Value error encountered: {val_err}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="librosa", choices=["librosa", "opensmile", "wav2vec"])
    parser.add_argument("--data_path", type=str, default="datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1/audio_speech_actors_01-24")
    args = parser.parse_args()
    
    pipeline = SERTrainingPipeline(feature_type=args.feature_type)
    pipeline.run_training_pipeline(args.data_path)
