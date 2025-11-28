from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model parameters
MAX_FEATURES = 5000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Sentiment labels
SENTIMENT_LABELS = {
    0: "Negative",
    1: "Positive"
}

# Color scheme for visualization
COLORS = {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#FFC107"
}