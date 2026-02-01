import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://leduc:leduc@db:5432/leduc")

# Training artifacts
TRAINING_DIR = Path(os.getenv("LEDUC_TRAINING_DIR", BASE_DIR.parent / ".." / "leduc-bot-training")).resolve()
MODEL_PATH = Path(os.getenv("LEDUC_MODEL_PATH", TRAINING_DIR / "policy_net_leduc.pt")).resolve()

# API behavior
DEFAULT_BOT_PLAYER = int(os.getenv("DEFAULT_BOT_PLAYER", "1"))

# CORS
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,https://leducpokerbot.vercel.app",
).split(",")
