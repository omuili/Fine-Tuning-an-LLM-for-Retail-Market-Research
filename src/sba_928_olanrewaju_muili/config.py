"""Project paths and experiment settings."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_DIR / "data" / "online+retail.zip"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_DIR = PROJECT_DIR / "models" / "fine_tuned_model"

MODEL_NAME = "google/flan-t5-small"

EPOCHS = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SEED = 42

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64

# Prompt dataset size.
NUM_COUNTRIES = 20
NUM_PRODUCTS = 30
NUM_COMPARED_PRODUCTS = 20
TRAIN_FRACTION = 0.8
