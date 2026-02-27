from ser_project.logger import Logging  # Use the full path from src
import pickle
from ser_project.training.data_ingestion import SERDataLoader
from ser_project.training.training import build_ser_cnn

logger = Logging()
class SERTrainingPipeline:
    def run_training_pipeline(self, raw_data_path):
        try:
            logger.info("--- Phase 1: Ingesting Raw Audio ---")
            loader = SERDataLoader(raw_data_path)
            x_raw, y_raw = loader.process_dataset()

            logger.info("--- Phase 2: Preprocessing and Reshaping ---")
            x_train, x_test, y_train, y_test, scaler = loader.prepare_pipeline_data(x_raw, y_raw)

            logger.info("--- Phase 3: Training Model ---")
            model = build_ser_cnn(input_shape=(x_train.shape[1], 1))
            history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

            logger.info("--- Phase 4: Evaluation ---")
            loss, acc = model.evaluate(x_test, y_test)
            logger.info(f"Pipeline Test Accuracy: {acc:.4f}")

            logger.info("--- Phase 5: Saving Pipeline Artifacts ---")
            try:
                model.save('ser_project/artifacts/ser_model.keras')
                with open('ser_project/artifacts/scaler.pkl', 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info("Pipeline execution complete.")
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

