"""Configuration for KrillLM Telegram Manager."""

import os

# Telegram bot token — set via environment variable
BOT_TOKEN: str = os.environ.get("KRILLM_TG_TOKEN", "")

# Comma-separated Telegram user IDs allowed to use the bot.
# REQUIRED — the bot refuses all users when this is not set.
ALLOWED_USERS: list[int] = [
    int(uid.strip())
    for uid in os.environ.get("KRILLM_TG_USERS", "").split(",")
    if uid.strip().isdigit()
]

# KrillLM server settings
KRILLM_HOST: str = os.environ.get("KRILLM_HOST", "127.0.0.1")
KRILLM_PORT: int = int(os.environ.get("KRILLM_PORT", "57455"))
KRILLM_BASE_URL: str = f"http://{KRILLM_HOST}:{KRILLM_PORT}"

# Path to the krillm CLI binary
KRILLM_BIN: str = os.environ.get("KRILLM_BIN", "krillm")
