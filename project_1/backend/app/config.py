"""Application configuration and settings."""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ML_MODELS_DIR = BASE_DIR / "app" / "ml" / "models"
DATASET_PATH = PROJECT_ROOT / "creditcard.csv"
DATASET_SAMPLE_PATH = DATA_DIR / "creditcard_sample.csv"

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR / 'fraud_detection.db'}"

# Auth
SECRET_KEY = os.getenv("SECRET_KEY", "info812-fraud-detection-secret-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480  # 8 hours

# Demo users (analyst and admin)
DEMO_USERS = {
    "analyst": {
        "username": "analyst",
        "full_name": "Fraud Analyst",
        "email": "analyst@ais.ac.nz",
        "hashed_password": "",  # Will be set at startup
        "role": "analyst",
    },
    "admin": {
        "username": "admin",
        "full_name": "Team Leader",
        "email": "admin@ais.ac.nz",
        "hashed_password": "",
        "role": "admin",
    },
}

# Cost matrix (from report Section 8.6)
DEFAULT_FN_COST = 250  # €250 per missed fraud
DEFAULT_FP_COST = 15   # €15 per false alarm

# Model scoring
GREY_ZONE_LOW = 0.40   # Below this → auto-allow
GREY_ZONE_HIGH = 0.75  # Above this → auto-flag
# Between low and high → run J48 rules for explainability

# Feature selection (top features from report Table 3)
SELECTED_FEATURES = [
    "V17", "V14", "V12", "V10", "V4", "V11",
    "V16", "V3", "V9", "V7", "Amount",
]

# All V features
ALL_V_FEATURES = [f"V{i}" for i in range(1, 29)]

# CORS
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
    "http://127.0.0.1:3000",
]
