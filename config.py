"""Central configuration: all paths are relative to the project root so the
project runs unchanged on Windows and Linux."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DB_PATH = Path(os.environ.get("FND_DB_PATH", ROOT / "database" / "detections.db"))
UPLOAD_DIR = ROOT / "app" / "static" / "uploads"

# Raw dataset locations
FAKENEWSNET_DIR = DATA_RAW / "fakenewsnet"
MEDIAEVAL_DIR = DATA_RAW / "mediaeval"

# Model artefacts
TEXT_NEWS_MODEL = MODELS_DIR / "text_news_tfidf_lr.joblib"        # FakeNewsNet headlines
TEXT_SOCIAL_MODEL = MODELS_DIR / "text_social_tfidf_lr.joblib"    # MediaEval tweets
IMAGE_CONTEXT_MODEL = MODELS_DIR / "image_context_effnet_lr.joblib"  # MediaEval images
MANIPULATION_MODEL = MODELS_DIR / "manipulation_forensic.joblib"  # synthetic tampering
AI_GENERATED_MODEL = MODELS_DIR / "ai_generated_effnet_lr.joblib"  # optional (CIFAKE)
FUSION_MODEL = MODELS_DIR / "fusion_lr.joblib"
CNN_WEIGHTS_DIR = MODELS_DIR / "cnn_weights"

RANDOM_SEED = 42
IMAGE_SIZE = 224
MAX_UPLOAD_MB = 100
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp", "gif"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

# Below this margin from 0.5 a modality score is reported as "uncertain".
UNCERTAIN_MARGIN = 0.15
