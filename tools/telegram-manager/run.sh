#!/usr/bin/env bash
# Quick start script for Krill Telegram Manager.
#
# Usage:
#   export KRILL_TG_TOKEN="your-bot-token"
#   export KRILL_TG_USERS="123456789"  # your Telegram user ID
#   ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${KRILL_TG_TOKEN:-}" ]; then
    echo "Error: KRILL_TG_TOKEN is not set."
    echo "Get a token from @BotFather on Telegram, then:"
    echo "  export KRILL_TG_TOKEN=\"your-token-here\""
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies..."
    .venv/bin/pip install -q -r requirements.txt
fi

echo "Starting Krill Telegram Manager..."
exec .venv/bin/python bot.py
