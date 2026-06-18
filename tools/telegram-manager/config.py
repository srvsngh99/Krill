"""Configuration for Krill Telegram Manager."""

import os

# Telegram bot token — set via environment variable
BOT_TOKEN: str = os.environ.get("KRILL_TG_TOKEN", "")

# Comma-separated Telegram user IDs allowed to use the bot.
# REQUIRED — the bot refuses all users when this is not set.
ALLOWED_USERS: list[int] = [
    int(uid.strip())
    for uid in os.environ.get("KRILL_TG_USERS", "").split(",")
    if uid.strip().isdigit()
]

# Krill server settings
KRILL_HOST: str = os.environ.get("KRILL_HOST", "127.0.0.1")
KRILL_PORT: int = int(os.environ.get("KRILL_PORT", "57455"))
KRILL_BASE_URL: str = f"http://{KRILL_HOST}:{KRILL_PORT}"

# Path to the krill CLI binary
KRILL_BIN: str = os.environ.get("KRILL_BIN", "krill")
