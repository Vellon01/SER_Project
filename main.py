from ser_project.training.train import SERTrainingPipeline
from ser_project.artifacts import SERDataLoaderArtifacts
sertrainer = SERTrainingPipeline()
if __name__ == "__main__":
    DATA_PATH = SERDataLoaderArtifacts.data_path
    sertrainer.run_training_pipeline(DATA_PATH)