from ser_project.training.model_evaluator import ModelEvaluator
from ser_project.training.train import SERTrainingPipeline
from ser_project.artifacts import SERDataLoaderArtifacts
sertrainer = SERTrainingPipeline()
evaluator = ModelEvaluator()

if __name__ == "__main__":
    DATA_PATH = SERDataLoaderArtifacts.data_path
    sertrainer.run_training_pipeline(DATA_PATH)
    evaluator.evaluate_and_update()