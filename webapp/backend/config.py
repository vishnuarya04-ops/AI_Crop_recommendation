"""
Configuration — loads settings from .env file.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the webapp directory
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Crop/

_model_env = os.getenv("MODEL_PATH", "models/random_forest.joblib")
MODEL_PATH = Path(_model_env)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BASE_DIR / MODEL_PATH).resolve()

_pipe_env = os.getenv("PIPELINE_PATH", "preprocessing_pipeline.joblib")
PIPELINE_PATH = Path(_pipe_env)
if not PIPELINE_PATH.is_absolute():
    PIPELINE_PATH = (BASE_DIR / PIPELINE_PATH).resolve()

# ── Server ─────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ── Logging ────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── API ────────────────────────────────────────────────
API_PREFIX = os.getenv("API_PREFIX", "/api")

# ── App Metadata ───────────────────────────────────────
APP_TITLE = "🌾 Crop Recommendation API"
APP_VERSION = "1.1.0"
APP_DESCRIPTION = (
    "Production REST API for AI-based crop recommendation. "
    "Accepts soil nutrient levels and climate conditions, "
    "returns the optimal crop with confidence scores. "
    "Built with FastAPI + Random Forest (99.55% accuracy)."
)
