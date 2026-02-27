import os
import shutil
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from ser_project.logger import Logging
from ser_project.training.data_ingestion import SERDataLoader
from ser_project.artifacts import SERDataLoaderArtifacts

logger = Logging()

class ModelEvaluator:
    def __init__(self):
        # Define paths within the artifacts directory
        self.latest_model_path = 'ser_project/artifacts/ser_model.keras'
        self.best_model_path = 'ser_project/artifacts/best_model.keras'
        self.archive_dir = 'ser_project/artifacts/archive'
        self.results_path = 'ser_project/artifacts/evaluation_results.json'
        
        # Ensure the archive directory exists
        os.makedirs(self.archive_dir, exist_ok=True)

    def get_evaluation_data(self):
        """
        Fetches the validation set using the existing data loader
        """
        logger.info("Fetching validation data for evaluation...")
        loader = SERDataLoader(SERDataLoaderArtifacts.data_path)
        x_raw, y_raw = loader.process_dataset()
        # We only need the validation split (x_val/y_val, which maps to x_test/y_test in prepare_pipeline_data)
        _, x_val, _, y_val, _ = loader.prepare_pipeline_data(x_raw, y_raw)
        return x_val, y_val

    def evaluate_and_update(self):
        # Load the latest trained model
        if not os.path.exists(self.latest_model_path):
            logger.error(f"Latest model not found at {self.latest_model_path}. Train a model first.")
            return

        logger.info("Loading the latest trained model...")
        latest_model = load_model(self.latest_model_path)

        # Fetch the validation set
        x_val, y_val = self.get_evaluation_data()

        # Check the accuracy of the model on the validation set
        logger.info("Evaluating the latest model...")
        latest_loss, latest_acc = latest_model.evaluate(x_val, y_val, verbose=0)
        logger.info(f"Latest Model Accuracy: {latest_acc:.4f}")

        # Prepare results dictionary
        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latest_model_accuracy": latest_acc,
            "latest_model_loss": latest_loss
        }

        # Load the old model and check the accuracy on the validation set
        best_acc = 0.0
        if os.path.exists(self.best_model_path):
            logger.info("Loading the current best model for comparison...")
            best_model = load_model(self.best_model_path)
            
            logger.info("Evaluating the current best model...")
            best_loss, best_acc = best_model.evaluate(x_val, y_val, verbose=0)
            logger.info(f"Current Best Model Accuracy: {best_acc:.4f}")
            
            results["best_model_accuracy"] = best_acc
            results["best_model_loss"] = best_loss
        else:
            logger.info("No existing best model found. The latest model will automatically become the best.")

        # Compare the results and see if there is an improvement
        if latest_acc > best_acc:
            logger.info("Improvement detected!")
            
            # Save the old model as an archived model
            if os.path.exists(self.best_model_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_model_path = os.path.join(self.archive_dir, f"model_{timestamp}.keras")
                shutil.move(self.best_model_path, archived_model_path)
                logger.info(f"Archived old best model to {archived_model_path}")
            
            # Save the new model as the best model
            latest_model.save(self.best_model_path)
            logger.info(f"Saved the latest model as the new best model at {self.best_model_path}")
            results["action"] = "latest_model_promoted"
        else:
            logger.info("The latest model did not improve upon the best model. No changes made to the best model.")
            results["action"] = "latest_model_discarded"

        # Save the results somewhere for later analysis
        with open(self.results_path, 'a') as f:
            f.write(json.dumps(results) + '\n')
        logger.info(f"Saved evaluation results to {self.results_path}")
