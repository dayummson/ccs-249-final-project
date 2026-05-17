# Project Analyzer using Fine tuned "distilbert-base-uncased"

A document analysis project that trains a DistilBERT classifier on synthetic rubric data and exposes a Flask API for extracting and classifying blocks from .docx or .pdf activity sheets.

## Repository overview

- backend/ - Flask API for document upload, extraction, and classification
- constants/ - label definitions and shared constants
- datasets/ - synthetic and misc training datasets
- final_requirement_model/ - packaged fine-tuned model artifacts for deployment
- graphs/ - training/analysis plots
- inferences/ - inference scripts for raw and fine-tuned models
- miscs/ - experiments, raw model scripts, and utility notes
- models/ - saved model checkpoints (alpha)
- outputs/ - generated briefing and inference outputs
- results/ - training checkpoints and trainer state
- samples/ - sample files used for local testing
- scripts/ - training, data generation, and evaluation utilities
- tests/ - lightweight scripts to validate extractors
- utils/ - extractors, metrics, and validators used by training and backend

## Key paths for new contributors

### Training

Primary training entry point:

- scripts/trainings/train.py - trains the fine-tuned DistilBERT classifier

Training output locations:

- models/alpha/ - final saved model (config, tokenizer, safetensors)
- results/ - Trainer checkpoints and state files
- scripts/trainings/results/ - legacy/alternate training output
- graphs/ - charts and metrics visualizations (if generated)

### Inference and outputs

- inferences/fine_tuned/ - inference scripts using fine-tuned model
- inferences/raw_model/ - inference scripts using base model
- outputs/ - briefing outputs and saved inference files
- miscs/raw_model.py - raw model inference and heuristic baseline

### Datasets and labels

- datasets/synthetic_data/ - main training dataset
- datasets/miscs/ - class-specific and curated datasets
- constants/label.py - label mappings and constants

## Setup

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

From the repo root:

```bash
python scripts/trainings/train.py
```

Expected artifacts:

- models/alpha/ (final model + tokenizer)
- results/ (checkpoint folders)

## Model comparison (branch)

On the `model-versions-comparison` branch, you can run the evaluation script at [scripts/evaluations/evaluate.py](scripts/evaluations/evaluate.py) to compare model quality. It shows that the initial model trained on the 5000-item dataset is outperformed by the model trained on the 1800-item quality dataset. Check that branch to see the metric gap and reports.

From the repo root on that branch:

```bash
python scripts/evaluations/evaluate.py
```

## Run the Flask backend

From the repo root:

```bash
python backend/app.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Analyze a document (multipart form upload):

```bash
curl -F "file=@samples/final-project.pdf" http://localhost:5000/analyze
```

## Test the extractors

```bash
python tests/pdf_extract.py
```

## Notes

- The classifier is built on top of the base model "distilbert-base-uncased".
- Uploads are validated and parsed through utils/ before classification.
- If you add new dataset files, keep them under datasets/ and update scripts/ accordingly.
