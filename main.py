from ser_project.training.train import SERTrainingPipeline
sertrainer = SERTrainingPipeline()
if __name__ == "__main__":
    DATA_PATH = "path/to/your/ravdess-audio"
    sertrainer.run_training_pipeline(DATA_PATH)