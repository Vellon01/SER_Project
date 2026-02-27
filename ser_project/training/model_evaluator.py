import os
import shutil
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from ser_project.logger import Logging

logger = Logging()

class ModelEvaluator:
    def __init__(self):
        self.latest_model_path = 'ser_project/artifacts/ser_model.keras'
        self.best_model_path = 'ser_project/artifacts/best_model.keras'
        self.archive_dir = 'ser_project/artifacts/archive'
        self.results_path = 'ser_project/artifacts/evaluation_results.json'
        
        os.makedirs(self.archive_dir, exist_ok=True)

    # Note: get_evaluation_data() has been completely removed

    # Accept x_val and y_val as arguments directly
    def evaluate_and_update(self, x_val, y_val):
        if not os.path.exists(self.latest_model_path):
            logger.error(f"Latest model not found at {self.latest_model_path}. Train a model first.")
            return

        logger.info("Loading the latest trained model...")
        latest_model = load_model(self.latest_model_path)

        # Removed the call to self.get_evaluation_data()
        
        logger.info("Evaluating the latest model...")
        latest_loss, latest_acc = latest_model.evaluate(x_val, y_val, verbose=0)
        logger.info(f"Latest Model Accuracy: {latest_acc:.4f}")

        results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "latest_model_accuracy": latest_acc,
            "latest_model_loss": latest_loss
        }

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

        if latest_acc > best_acc:
            logger.info("Improvement detected!")
            if os.path.exists(self.best_model_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archived_model_path = os.path.join(self.archive_dir, f"model_{timestamp}.keras")
                shutil.move(self.best_model_path, archived_model_path)
                logger.info(f"Archived old best model to {archived_model_path}")
            
            latest_model.save(self.best_model_path)
            logger.info(f"Saved the latest model as the new best model at {self.best_model_path}")
            results["action"] = "latest_model_promoted"
        else:
            logger.info("The latest model did not improve upon the best model. No changes made to the best model.")
            results["action"] = "latest_model_discarded"

        with open(self.results_path, 'a') as f:
            f.write(json.dumps(results) + '\n')
        logger.info(f"Saved evaluation results to {self.results_path}")