"""
WhatsApp AI Bot - FastAPI + Groq + Reminders + Meta WhatsApp Cloud API

.env file:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
WHATSAPP_TOKEN=EAAxxxxxxxxxxxxxxxxxxxxx
PHONE_NUMBER_ID=1200795349780072
VERIFY_TOKEN=my_secret_token_123
"""

import os
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import requests

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
WHATSAPP_TOKEN  = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN    = os.getenv("VERIFY_TOKEN", "my_secret_token_123")
GROQ_MODEL      = "llama-3.3-70b-versatile"
GRAPH_API_VERSION = "v25.0"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whatsapp-bot")

logger.info("=" * 60)
logger.info("STARTUP DIAGNOSTICS")
logger.info("  VERIFY_TOKEN    : %s", VERIFY_TOKEN)
logger.info("  PHONE_NUMBER_ID : %s", PHONE_NUMBER_ID or "(MISSING ❌)")
logger.info("  WHATSAPP_TOKEN  : %s", "SET ✅" if WHATSAPP_TOKEN else "(MISSING ❌)")
logger.info("  GROQ_API_KEY    : %s", "SET ✅" if GROQ_API_KEY else "(MISSING ❌)")
logger.info("  GROQ_MODEL      : %s", GROQ_MODEL)
logger.info("  CURRENT IST     : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
logger.info("=" * 60)

# ── Import Groq ───────────────────────────────────────────────────────────────
groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq SDK : OK ✅")
except ImportError:
    logger.error("Groq SDK NOT installed! Run: pip install groq")

# ── Conversation Memory ───────────────────────────────────────────────────────
conversation_history: Dict[str, list] = {}

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="WhatsApp AI Bot with Reminders")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE — SQLite for reminders
# All times stored as IST in "YYYY-MM-DD HH:MM:SS" format
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE = "reminders.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            task TEXT NOT NULL,
            remind_at TEXT NOT NULL,
            sent INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized ✅")

def save_reminder(phone: str, task: str, remind_at: datetime):
    """
    Save reminder. remind_at must be a datetime (naive or aware).
    We always store as IST string.
    """
    # If naive, assume it's already IST
    if remind_at.tzinfo is None:
        remind_at_ist = remind_at.replace(tzinfo=IST)
    else:
        remind_at_ist = remind_at.astimezone(IST)

    remind_at_str = remind_at_ist.strftime("%Y-%m-%d %H:%M:%S")

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO reminders (phone, task, remind_at) VALUES (?, ?, ?)",
        (phone, task, remind_at_str)
    )
    conn.commit()
    conn.close()
    logger.info("[DB] Reminder saved: '%s' at %s (IST) for %s", task, remind_at_str, phone)

def get_due_reminders():
    """
    Get all reminders due at or before right now (IST).
    Compares full datetime strings — no truncation bugs.
    """
    conn = sqlite3.connect(DB_FILE)
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    logger.debug("[SCHEDULER] Checking due reminders. Now (IST): %s", now_str)
    rows = conn.execute(
        "SELECT id, phone, task, remind_at FROM reminders WHERE remind_at <= ? AND sent = 0",
        (now_str,)
    ).fetchall()
    conn.close()
    return rows

def mark_sent(reminder_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()


# ─────────────────────────────────────────────────────────────────────────────
# REMINDER SCHEDULER — background thread, runs every 30 seconds
# ─────────────────────────────────────────────────────────────────────────────
def reminder_scheduler():
    logger.info("[SCHEDULER] Started ✅")
    while True:
        try:
            due = get_due_reminders()
            if due:
                logger.info("[SCHEDULER] Found %d due reminder(s)", len(due))
            for reminder_id, phone, task, remind_at in due:
                try:
                    send_whatsapp_text(
                        phone_number=phone,
                        message=f"⏰ *Reminder:* {task}"
                    )
                    mark_sent(reminder_id)
                    logger.info("[SCHEDULER] ✅ Sent reminder #%d to %s: %s", reminder_id, phone, task)
                except Exception as e:
                    logger.error("[SCHEDULER] ❌ Failed to send reminder #%d: %s", reminder_id, e)
        except Exception as e:
            logger.error("[SCHEDULER] ❌ Loop error: %s", e)
        time.sleep(30)

scheduler_thread = threading.Thread(target=reminder_scheduler, daemon=True)
scheduler_thread.start()
logger.info("[SCHEDULER] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# PARSE REMINDER USING AI
# ─────────────────────────────────────────────────────────────────────────────
def parse_reminder_with_ai(message: str) -> dict | None:
    """
    Use Groq AI to parse natural language reminder.
    Returns dict with 'task' and 'remind_at' (YYYY-MM-DD HH:MM) or None.
    """
    try:
        now_ist = datetime.now(IST)
        now_str = now_ist.strftime("%Y-%m-%d %H:%M:%S")

        prompt = f"""Current date and time (IST, Asia/Kolkata): {now_str}

User message: "{message}"

Is this a reminder request? Extract the details.

Reply ONLY in this exact JSON format with absolutely no other text, no markdown, no explanation:

If it IS a reminder:
{{"is_reminder": true, "task": "the task description", "remind_at": "YYYY-MM-DD HH:MM"}}

If it is NOT a reminder:
{{"is_reminder": false}}

Rules:
- Use 24-hour time format for remind_at
- Calculate the exact date/time for relative phrases like "tomorrow", "in 2 hours", "next Monday", "at 5pm" based on the current IST time above
- The remind_at must be in the future relative to the current time
- task should be a clean description of what to remind about"""

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()
        # Strip any markdown code fences
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        logger.info("[REMINDER_PARSER] Parsed: %s", data)
        return data

    except json.JSONDecodeError as e:
        logger.error("[REMINDER_PARSER] JSON parse error: %s | Raw: %s", e, text)
        return None
    except Exception as e:
        logger.error("[REMINDER_PARSER] Error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AI REPLY USING GROQ
# ─────────────────────────────────────────────────────────────────────────────
def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] Message from %s: %s", sender, prompt[:120])

    if not GROQ_API_KEY:
        logger.error("[AI] GROQ_API_KEY missing")
        return "Sorry, AI is not configured."

    if groq_client is None:
        logger.error("[AI] Groq SDK not installed")
        return "Sorry, AI SDK is missing."

    try:
        # ── Check if this is a reminder request ──────────────────────────────
        reminder_keywords = [
            "remind", "reminder", "alert", "notify", "don't forget",
            "याद", "याद दिलाओ", "रिमाइंडर", "set a reminder", "set reminder"
        ]
        is_likely_reminder = any(kw in prompt.lower() for kw in reminder_keywords)

        if is_likely_reminder:
            logger.info("[AI] Detected possible reminder request")
            reminder_data = parse_reminder_with_ai(prompt)

            if reminder_data and reminder_data.get("is_reminder"):
                try:
                    task = reminder_data.get("task", prompt)
                    remind_at_str = reminder_data.get("remind_at")  # "YYYY-MM-DD HH:MM"

                    # Parse as naive datetime, then attach IST timezone
                    remind_at_naive = datetime.strptime(remind_at_str, "%Y-%m-%d %H:%M")
                    remind_at_ist   = remind_at_naive.replace(tzinfo=IST)

                    # Validate it's in the future
                    now_ist = datetime.now(IST)
                    if remind_at_ist <= now_ist:
                        logger.warning("[AI] Reminder time is in the past: %s", remind_at_ist)
                        return (
                            "⚠️ That time seems to be in the past! "
                            "Please give me a future time for the reminder."
                        )

                    save_reminder(phone=sender, task=task, remind_at=remind_at_ist)

                    formatted = remind_at_ist.strftime("%d %b %Y at %I:%M %p IST")
                    logger.info("[AI] ✅ Reminder saved for %s", formatted)
                    return (
                        f"✅ *Reminder set!*\n"
                        f"📋 Task: {task}\n"
                        f"⏰ Time: {formatted}\n\n"
                        f"I'll remind you then!"
                    )

                except ValueError as e:
                    logger.error("[AI] Failed to parse reminder datetime: %s", e)
                    return (
                        "Sorry, I couldn't understand that time. "
                        "Try something like: _'Remind me tomorrow at 5pm to call John'_"
                    )
                except Exception as e:
                    logger.error("[AI] Failed to save reminder: %s", e)
                    return "Sorry, I couldn't set that reminder. Please try again."

        # ── Normal AI reply ───────────────────────────────────────────────────
        if sender not in conversation_history:
            conversation_history[sender] = []

        conversation_history[sender].append({
            "role": "user",
            "content": prompt
        })

        # Keep last 10 messages for context
        recent = conversation_history[sender][-10:]

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful and friendly WhatsApp assistant.\n"
                        "- Keep replies short and clear (under 100 words)\n"
                        "- Be conversational and warm\n"
                        "- Reply in the same language the user writes in\n"
                        "- You can set reminders when asked\n"
                        "- Never make up false information"
                    )
                },
                *recent
            ],
            max_tokens=300,
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        logger.info("[AI] ✅ Reply: %s", reply[:200])

        conversation_history[sender].append({
            "role": "assistant",
            "content": reply
        })

        return reply

    except Exception as e:
        logger.exception("[AI] Error: %s", e)
        return "Sorry, I encountered an error. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# SEND WHATSAPP MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp_text(
    phone_number: str,
    message: str,
    phone_number_id: str | None = None,
) -> dict:
    effective_phone_id = phone_number_id or PHONE_NUMBER_ID

    if not WHATSAPP_TOKEN:
        raise ValueError("WHATSAPP_TOKEN missing in .env")
    if not effective_phone_id:
        raise ValueError("PHONE_NUMBER_ID missing in .env")

    url = (
        f"https://graph.facebook.com/{GRAPH_API_VERSION}"
        f"/{effective_phone_id}/messages"
    )
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }

    logger.info("[SEND] Sending to %s...", phone_number)
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    logger.info("[SEND] Status: %s", resp.status_code)

    if not resp.ok:
        raise RuntimeError(f"Meta API error {resp.status_code}: {resp.text}")

    logger.info("[SEND] ✅ Message sent!")
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    params       = dict(request.query_params)
    mode         = params.get("hub.mode") or params.get("mode")
    verify_token = params.get("hub.verify_token") or params.get("verify_token")
    challenge    = params.get("hub.challenge") or params.get("challenge")

    if verify_token == VERIFY_TOKEN and challenge:
        logger.info("[VERIFY] ✅ Webhook verified!")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("[VERIFY] ❌ Token mismatch. Got: %s", verify_token)
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# RECEIVE MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(payload: Dict[str, Any]):
    logger.info("[WEBHOOK] Payload received")

    try:
        entries = payload.get("entry", [])
        if not entries:
            return {"status": "ignored"}

        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})

                if "statuses" in value and "messages" not in value:
                    logger.info("[WEBHOOK] Status update — skipping")
                    continue

                messages = value.get("messages") or []
                if not messages:
                    continue

                metadata_phone_id = value.get("metadata", {}).get("phone_number_id")
                phone_id_to_use   = metadata_phone_id or PHONE_NUMBER_ID

                for message in messages:
                    try:
                        sender   = message.get("from")
                        msg_type = message.get("type")

                        if not sender:
                            continue

                        if msg_type == "text":
                            text_body = message.get("text", {}).get("body", "")
                            logger.info("[MSG] from=%s text=%s", sender, text_body)
                            reply = generate_ai_reply(text_body, sender)
                        elif msg_type == "image":
                            reply = "I received your image! I can only process text for now. 😊"
                        elif msg_type == "audio":
                            reply = "I received your voice message! I can only process text for now. 😊"
                        elif msg_type == "document":
                            reply = "I received your document! I can only process text for now. 😊"
                        else:
                            continue

                        try:
                            send_whatsapp_text(
                                phone_number=sender,
                                message=reply,
                                phone_number_id=phone_id_to_use,
                            )
                        except ValueError as e:
                            logger.error("[MSG] Config error: %s", e)
                        except RuntimeError as e:
                            logger.error("[MSG] WhatsApp API error: %s", e)
                            if "401" in str(e):
                                logger.error(
                                    "[MSG] ❌ Token expired! "
                                    "Regenerate at developers.facebook.com"
                                )
                        except Exception as e:
                            logger.exception("[MSG] Send error: %s", e)

                    except Exception as e:
                        logger.exception("[MSG] Processing error: %s", e)

    except Exception as e:
        logger.exception("[WEBHOOK] Error: %s", e)

    return {"status": "received"}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    conn = sqlite3.connect(DB_FILE)
    pending = conn.execute(
        "SELECT COUNT(*) FROM reminders WHERE sent = 0"
    ).fetchone()[0]
    total = conn.execute(
        "SELECT COUNT(*) FROM reminders"
    ).fetchone()[0]
    # Show next 5 pending reminders for debugging
    upcoming = conn.execute(
        "SELECT phone, task, remind_at FROM reminders WHERE sent = 0 ORDER BY remind_at LIMIT 5"
    ).fetchall()
    conn.close()

    return {
        "status"            : "ok ✅",
        "current_time_ist"  : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "groq_api_key"      : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "whatsapp_token"    : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "phone_number_id"   : PHONE_NUMBER_ID or "MISSING ❌",
        "groq_sdk"          : "OK ✅" if groq_client else "ERROR ❌",
        "scheduler"         : "RUNNING ✅",
        "reminders_pending" : pending,
        "reminders_total"   : total,
        "upcoming_reminders": [
            {"phone": r[0], "task": r[1], "remind_at": r[2]}
            for r in upcoming
        ],
        "users_in_memory"   : len(conversation_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)