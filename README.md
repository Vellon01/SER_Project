# SER_Project

Speech Emotion Recognition (SER) — an end-to-end Python project for
building, training and evaluating emotion-recognition models from speech.

This repository contains data ingestion, feature extraction, training and
evaluation scripts and a notebook for experimentation. The included dataset
directory references the RAVDESS dataset under `datasets/uwrfkaggler`.

Key components
- `main.py`: Lightweight entry point for running scripts or experiments.
- `SER.ipynb`: Notebook for interactive exploration and experiments.
- `ser_project/training/data_ingestion.py`: dataset loading and parsing logic.
- `ser_project/training/features.py`: audio feature extraction (MFCCs, etc.).
- `ser_project/training/train.py`: training script to train models from features.
- `ser_project/training/model_evaluator.py`: evaluation and metrics utilities.
- `ser_project/training/training.py`: higher-level training orchestration.
- `ser_project/artifacts/`: directory where trained models and artifacts are stored.
- `logs/`: runtime logs and training logs.

Dataset
- The project expects the RAVDESS audio files under:
	`datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1/audio_speech_actors_01-24/`
- Keep the original folder structure (actors under `Actor_01` ... `Actor_24`).

Installation
1. Create a Python virtual environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
# or install editable package if you plan to iterate on the code
pip install -e .
```

Quick start
- Run the notebook for exploration: open `SER.ipynb` in Jupyter.
- To train a model from the command line (example):

```bash
python ser_project/training/train.py --config config/train.yaml
```

(Replace the command above with the exact CLI/config options you use.)

Running evaluation
- After training, evaluate a saved model using the evaluator utilities:

```bash
python -m ser_project.training.model_evaluator --model ser_project/artifacts/model.pkl
```

Project structure (top-level)
- `datasets/` – audio dataset (RAVDESS) and any dataset manifests.
- `ser_project/` – implementation package (training, utils, constants).
- `logs/` – runtime and training logs.
- `requirements.txt`, `pyproject.toml`, `setup.py` – dependency and packaging files.

Notes
- Ensure the dataset path is present before running ingestion or training.
- The codebase is structured so key steps can be reused in experiments and pipelines.

Contributing
- Feel free to open issues or pull requests. Describe reproducible steps and
	include environment details when reporting bugs.

Contact
- Author: see repository owner. Use the repository issue tracker for questions.

License
- Check `setup.py` / project metadata for license details, or add a `LICENSE` file.

