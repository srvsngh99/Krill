#!/usr/bin/env python3
"""Krill Telegram Manager — control your local LLM from Telegram."""

from __future__ import annotations

import html
import logging
import sys
import textwrap

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode

from config import BOT_TOKEN, ALLOWED_USERS
from krill_client import KrillClient

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("krill-tg")

client = KrillClient()

# ── Auth decorator ───────────────────────────────────────────────────


def authorized(func):
    """Only allow configured user IDs. Denies all if ALLOWED_USERS is empty."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not ALLOWED_USERS:
            await update.message.reply_text(
                "Bot not configured. Set KRILL_TG_USERS to your Telegram user ID."
            )
            return
        if user is None or user.id not in ALLOWED_USERS:
            await update.message.reply_text("Not authorized.")
            return
        return await func(update, context)
    wrapper.__name__ = func.__name__
    return wrapper


# ── /start & /help ───────────────────────────────────────────────────


@authorized
async def cmd_start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        textwrap.dedent("""\
        <b>Krill Manager</b>

        <b>Model management</b>
        /models — list installed models
        /load &lt;name&gt; — load a model into memory
        /unload — unload the current model
        /pull &lt;name&gt; — download a model
        /rm &lt;name&gt; — delete a model

        <b>Inference</b>
        /chat &lt;prompt&gt; — single-shot chat
        /stream &lt;prompt&gt; — streaming chat (edits message live)

        <b>System</b>
        /status — server status &amp; resource usage
        /health — quick health check
        /bench &lt;name&gt; — run inference benchmark
        """),
        parse_mode=ParseMode.HTML,
    )


# ── Model management ────────────────────────────────────────────────


@authorized
async def cmd_models(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        models = await client.list_models()
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        return

    if not models:
        await update.message.reply_text("No models installed. Use /pull <name> to download one.")
        return

    lines = ["<b>Installed models:</b>"]
    for m in models:
        lines.append(f"  <code>{html.escape(m['id'])}</code>")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


@authorized
async def cmd_load(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /load <model-name>")
        return

    model = context.args[0]
    msg = await update.message.reply_text(f"Loading <code>{html.escape(model)}</code>...", parse_mode=ParseMode.HTML)

    try:
        result = await client.load_model(model)
        status = result.get("status", "unknown")
        family = result.get("family", "?")
        await msg.edit_text(
            f"Loaded <code>{html.escape(model)}</code> ({family})\nStatus: {status}",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await msg.edit_text(f"Failed to load: {e}")


@authorized
async def cmd_unload(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        await client.unload_model()
        await update.message.reply_text("Model unloaded.")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@authorized
async def cmd_pull(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /pull <model-name>")
        return

    model = context.args[0]
    msg = await update.message.reply_text(f"Pulling <code>{html.escape(model)}</code>...", parse_mode=ParseMode.HTML)

    last_text = ""
    try:
        async for line in client.pull_model(model):
            # Throttle edits — only update when content changed meaningfully
            if line and line != last_text:
                last_text = line
                try:
                    await msg.edit_text(
                        f"<code>{html.escape(line[-200:])}</code>",
                        parse_mode=ParseMode.HTML,
                    )
                except Exception:
                    pass  # telegram rate-limits edits
        await msg.edit_text(f"Pull complete: <code>{html.escape(model)}</code>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"Pull failed: {e}")


@authorized
async def cmd_rm(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /rm <model-name>")
        return

    model = context.args[0]
    keyboard = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Yes, delete", callback_data=f"rm_confirm:{model}"),
            InlineKeyboardButton("Cancel", callback_data="rm_cancel"),
        ]
    ])
    await update.message.reply_text(
        f"Delete model <code>{html.escape(model)}</code>? This cannot be undone.",
        parse_mode=ParseMode.HTML,
        reply_markup=keyboard,
    )


async def handle_rm_callback(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "rm_cancel":
        await query.edit_message_text("Cancelled.")
        return

    if query.data and query.data.startswith("rm_confirm:"):
        model = query.data.split(":", 1)[1]
        try:
            result = await client.remove_model(model)
            await query.edit_message_text(
                f"<code>{html.escape(result or 'Removed.')}</code>",
                parse_mode=ParseMode.HTML,
            )
        except Exception as e:
            await query.edit_message_text(f"Error: {e}")


# ── Inference ────────────────────────────────────────────────────────


@authorized
async def cmd_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /chat <your prompt>")
        return

    prompt = " ".join(context.args)
    msg = await update.message.reply_text("Thinking...")

    try:
        reply = await client.chat(prompt)
        # Telegram message limit is 4096 chars
        for i in range(0, len(reply), 4000):
            chunk = reply[i : i + 4000]
            if i == 0:
                await msg.edit_text(chunk)
            else:
                await update.message.reply_text(chunk)
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


@authorized
async def cmd_stream(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /stream <your prompt>")
        return

    prompt = " ".join(context.args)
    msg = await update.message.reply_text("...")

    accumulated = ""
    edit_counter = 0
    try:
        async for token in client.chat_stream(prompt):
            accumulated += token
            edit_counter += 1
            # Edit every 8 tokens to stay under Telegram rate limits
            if edit_counter % 8 == 0:
                try:
                    display = accumulated[-3500:]  # keep under message limit
                    await msg.edit_text(display or "...")
                except Exception:
                    pass
        # Final edit with full text
        if accumulated:
            for i in range(0, len(accumulated), 4000):
                chunk = accumulated[i : i + 4000]
                if i == 0:
                    await msg.edit_text(chunk)
                else:
                    await update.message.reply_text(chunk)
        else:
            await msg.edit_text("(empty response)")
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


# ── System ───────────────────────────────────────────────────────────


@authorized
async def cmd_status(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        s = await client.status()
    except Exception as e:
        await update.message.reply_text(f"Server unreachable: {e}")
        return

    lines = [
        f"<b>Krill Status</b>",
        f"State: <code>{s.get('status', '?')}</code>",
        f"Version: <code>{s.get('version', '?')}</code>",
        f"Memory: <code>{s.get('memory_mb', '?')} MB</code>",
        f"Uptime: <code>{s.get('uptime_seconds', '?')}s</code>",
    ]
    if s.get("model_loaded"):
        lines.append(f"Model: <code>{html.escape(str(s.get('model', '?')))}</code>")
        lines.append(f"Family: <code>{html.escape(str(s.get('family', '?')))}</code>")
        if mt := s.get("model_uptime_seconds"):
            lines.append(f"Model uptime: <code>{mt}s</code>")

    installed = s.get("installed_models", [])
    if installed:
        lines.append(f"\n<b>Installed ({len(installed)}):</b>")
        for name in installed:
            lines.append(f"  <code>{html.escape(name)}</code>")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


@authorized
async def cmd_health(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        h = await client.health()
        loaded = h.get("model_loaded", False)
        model = h.get("model", "none")
        await update.message.reply_text(
            f"Status: <code>{h.get('status', '?')}</code>\n"
            f"Model loaded: <code>{loaded}</code> ({html.escape(str(model))})",
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        await update.message.reply_text(f"Server down: {e}")


@authorized
async def cmd_bench(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /bench <model-name>")
        return

    model = context.args[0]
    msg = await update.message.reply_text(f"Benchmarking <code>{html.escape(model)}</code>...", parse_mode=ParseMode.HTML)

    try:
        result = await client.bench_model(model)
        await msg.edit_text(f"<pre>{html.escape(result[:4000])}</pre>", parse_mode=ParseMode.HTML)
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


# ── Fallback: plain text → chat ─────────────────────────────────────


@authorized
async def handle_text(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    """Treat plain text messages as chat prompts."""
    prompt = update.message.text
    if not prompt:
        return

    msg = await update.message.reply_text("...")

    accumulated = ""
    edit_counter = 0
    try:
        async for token in client.chat_stream(prompt):
            accumulated += token
            edit_counter += 1
            if edit_counter % 8 == 0:
                try:
                    await msg.edit_text(accumulated[-3500:] or "...")
                except Exception:
                    pass
        if accumulated:
            for i in range(0, len(accumulated), 4000):
                chunk = accumulated[i : i + 4000]
                if i == 0:
                    await msg.edit_text(chunk)
                else:
                    await update.message.reply_text(chunk)
        else:
            await msg.edit_text("(empty response)")
    except Exception as e:
        await msg.edit_text(f"Error: {e}")


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    if not BOT_TOKEN:
        print("Set KRILL_TG_TOKEN environment variable to your Telegram bot token.")
        print("Get one from @BotFather on Telegram.")
        sys.exit(1)

    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("models", cmd_models))
    app.add_handler(CommandHandler("load", cmd_load))
    app.add_handler(CommandHandler("unload", cmd_unload))
    app.add_handler(CommandHandler("pull", cmd_pull))
    app.add_handler(CommandHandler("rm", cmd_rm))
    app.add_handler(CommandHandler("chat", cmd_chat))
    app.add_handler(CommandHandler("stream", cmd_stream))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("bench", cmd_bench))

    # Callback queries (e.g. /rm confirmation)
    app.add_handler(CallbackQueryHandler(handle_rm_callback, pattern=r"^rm_"))

    # Plain text → streaming chat
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Close the httpx client on shutdown
    async def shutdown(_: Application) -> None:
        await client.close()
    app.post_shutdown = shutdown

    log.info("Krill Telegram Manager starting...")
    if ALLOWED_USERS:
        log.info("Authorized users: %s", ALLOWED_USERS)
    else:
        log.warning("No KRILL_TG_USERS set — bot will deny all users!")

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
