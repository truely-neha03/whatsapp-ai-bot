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
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import requests

# gTTS import — checked after logger is ready (see below)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN      = os.getenv("VERIFY_TOKEN", "my_secret_token_123")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GRAPH_API_VERSION = "v22.0"          # ← stable version, v25 can 404

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whatsapp-bot")

logger.info("=" * 60)
logger.info("STARTUP DIAGNOSTICS")
logger.info("  VERIFY_TOKEN    : %s", VERIFY_TOKEN)
logger.info("  PHONE_NUMBER_ID : %s", PHONE_NUMBER_ID or "MISSING ❌")
logger.info("  WHATSAPP_TOKEN  : %s", "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌")
logger.info("  GROQ_API_KEY    : %s", "SET ✅" if GROQ_API_KEY else "MISSING ❌")
logger.info("  GROQ_MODEL      : %s", GROQ_MODEL)
logger.info("  CURRENT IST     : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
logger.info("=" * 60)

# ── Groq ──────────────────────────────────────────────────────────────────────
groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq SDK : OK ✅")
except ImportError:
    logger.error("Groq SDK NOT installed — run: pip install groq")

# ── gTTS (Text to Speech) ─────────────────────────────────────────────────────
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
    logger.info("gTTS     : OK ✅")
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS NOT installed — voice notes disabled. Run: pip install gtts")

# ── Conversation Memory ───────────────────────────────────────────────────────
conversation_history: Dict[str, list] = {}

app = FastAPI(title="WhatsApp AI Bot with Reminders")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE = "reminders.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            phone      TEXT NOT NULL,
            task       TEXT NOT NULL,
            remind_at  TEXT NOT NULL,
            sent       INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialised ✅")


def save_reminder(phone: str, task: str, remind_at: datetime):
    """Store reminder. remind_at may be naive (assumed IST) or aware."""
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=IST)
    else:
        remind_at = remind_at.astimezone(IST)

    remind_at_str = remind_at.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO reminders (phone, task, remind_at) VALUES (?, ?, ?)",
        (phone, task, remind_at_str),
    )
    conn.commit()
    conn.close()
    logger.info("[DB] Saved reminder: '%s' at %s for %s", task, remind_at_str, phone)


def get_due_reminders():
    """Return reminders whose remind_at <= now (IST), not yet sent."""
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    conn    = sqlite3.connect(DB_FILE)
    rows    = conn.execute(
        "SELECT id, phone, task, remind_at FROM reminders "
        "WHERE remind_at <= ? AND sent = 0 ORDER BY remind_at",
        (now_str,),
    ).fetchall()
    conn.close()
    return rows


def mark_sent(reminder_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()


init_db()


# ─────────────────────────────────────────────────────────────────────────────
# SEND WHATSAPP MESSAGE  (defined BEFORE scheduler so it's available)
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp_text(phone_number: str, message: str, phone_number_id: str = "") -> dict:
    pid   = phone_number_id or PHONE_NUMBER_ID
    token = WHATSAPP_TOKEN

    # ── guard-rails ──────────────────────────────────────────────────────────
    if not token:
        raise ValueError("WHATSAPP_TOKEN is empty — check your .env")
    if not pid:
        raise ValueError("PHONE_NUMBER_ID is empty — check your .env")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to":               phone_number,
        "type":             "text",
        "text":             {"preview_url": False, "body": message},
    }

    logger.info("[SEND] POST %s  to=%s", url, phone_number)
    logger.info("[SEND] Token prefix: %s...", token[:12])

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error sending WhatsApp message: {e}") from e

    logger.info("[SEND] HTTP %s  body=%s", resp.status_code, resp.text[:300])

    if resp.status_code == 401:
        raise RuntimeError(
            "401 Unauthorised — your WHATSAPP_TOKEN has expired. "
            "Regenerate it at developers.facebook.com → your app → WhatsApp → API Setup."
        )
    if resp.status_code == 400:
        raise RuntimeError(f"400 Bad Request — {resp.text}")
    if not resp.ok:
        raise RuntimeError(f"Meta API {resp.status_code}: {resp.text}")

    logger.info("[SEND] ✅ Delivered to %s", phone_number)
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# VOICE NOTE — generate mp3 and send as WhatsApp audio
# ─────────────────────────────────────────────────────────────────────────────
def generate_voice_note(text: str) -> str | None:
    """
    Convert text → mp3 using gTTS.
    Returns the temp file path, or None if gTTS not available.
    """
    if not GTTS_AVAILABLE:
        logger.warning("[VOICE] gTTS not installed — skipping voice note")
        return None
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        logger.info("[VOICE] ✅ Generated voice note: %s", tmp.name)
        return tmp.name
    except Exception as exc:
        logger.error("[VOICE] ❌ gTTS error: %s", exc)
        return None


def send_whatsapp_audio(phone_number: str, audio_path: str, phone_number_id: str = "") -> dict:
    """
    Upload mp3 to Meta media endpoint, then send as WhatsApp audio message.
    """
    pid   = phone_number_id or PHONE_NUMBER_ID
    token = WHATSAPP_TOKEN

    if not token or not pid:
        raise ValueError("WHATSAPP_TOKEN or PHONE_NUMBER_ID missing")

    # ── Step 1: Upload audio file to Meta ────────────────────────────────────
    upload_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/media"
    with open(audio_path, "rb") as f:
        upload_resp = requests.post(
            upload_url,
            headers={"Authorization": f"Bearer {token}"},
            files={"file": ("reminder.mp3", f, "audio/mpeg")},
            data={"messaging_product": "whatsapp"},
            timeout=30,
        )

    if not upload_resp.ok:
        raise RuntimeError(f"Media upload failed {upload_resp.status_code}: {upload_resp.text}")

    media_id = upload_resp.json().get("id")
    logger.info("[VOICE] ✅ Uploaded media_id: %s", media_id)

    # ── Step 2: Send audio message using media_id ─────────────────────────────
    send_url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
    payload  = {
        "messaging_product": "whatsapp",
        "to":                phone_number,
        "type":              "audio",
        "audio":             {"id": media_id},
    }
    send_resp = requests.post(
        send_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        json=payload,
        timeout=15,
    )

    if not send_resp.ok:
        raise RuntimeError(f"Audio send failed {send_resp.status_code}: {send_resp.text}")

    logger.info("[VOICE] ✅ Audio message sent to %s", phone_number)

    # ── Cleanup temp file ─────────────────────────────────────────────────────
    try:
        os.remove(audio_path)
    except Exception:
        pass

    return send_resp.json()


def send_reminder_with_voice(phone: str, task: str):
    """
    Send reminder as BOTH a text message AND a voice note.
    If voice note fails, text message still goes through.
    """
    # 1. Always send text first
    send_whatsapp_text(
        phone_number=phone,
        message=f"⏰ *Reminder:* {task}",
    )
    logger.info("[REMINDER] ✅ Text sent to %s", phone)

    # 2. Try to send voice note
    voice_text = f"Hey! This is your reminder. {task}"
    audio_path = generate_voice_note(voice_text)
    if audio_path:
        try:
            send_whatsapp_audio(phone_number=phone, audio_path=audio_path)
            logger.info("[REMINDER] ✅ Voice note sent to %s", phone)
        except Exception as exc:
            logger.error("[REMINDER] ❌ Voice note failed (text was sent): %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# REMINDER SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
def reminder_scheduler():
    logger.info("[SCHEDULER] Thread started ✅")
    while True:
        try:
            due = get_due_reminders()
            if due:
                logger.info("[SCHEDULER] %d due reminder(s) found", len(due))

            for reminder_id, phone, task, remind_at in due:
                logger.info("[SCHEDULER] Firing reminder #%d → %s : %s", reminder_id, phone, task)
                try:
                    send_reminder_with_voice(phone=phone, task=task)
                    mark_sent(reminder_id)
                    logger.info("[SCHEDULER] ✅ Sent & marked done: #%d", reminder_id)
                except Exception as exc:
                    logger.error("[SCHEDULER] ❌ Failed reminder #%d: %s", reminder_id, exc)

        except Exception as exc:
            logger.error("[SCHEDULER] ❌ Loop error: %s", exc)

        time.sleep(30)


threading.Thread(target=reminder_scheduler, daemon=True).start()
logger.info("[SCHEDULER] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# KEEP-ALIVE — pings /health every 30s so Railway never sleeps
# ─────────────────────────────────────────────────────────────────────────────
RAILWAY_URL = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")

def keep_alive():
    # Wait 10s on startup before first ping
    time.sleep(10)
    logger.info("[KEEPALIVE] Thread started ✅  url=%s", RAILWAY_URL or "(not set yet)")
    while True:
        if RAILWAY_URL:
            try:
                url = f"https://{RAILWAY_URL}/health"
                resp = requests.get(url, timeout=10)
                logger.info("[KEEPALIVE] ✅ Pinged %s → %s", url, resp.status_code)
            except Exception as exc:
                logger.warning("[KEEPALIVE] ⚠️ Ping failed: %s", exc)
        else:
            logger.warning("[KEEPALIVE] ⚠️ RAILWAY_PUBLIC_DOMAIN not set — skipping ping")
        time.sleep(30)

threading.Thread(target=keep_alive, daemon=True).start()
logger.info("[KEEPALIVE] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# PARSE REMINDER WITH AI
# ─────────────────────────────────────────────────────────────────────────────
def parse_reminder_with_ai(message: str) -> dict | None:
    try:
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        prompt  = f"""Current date and time (IST): {now_str}

User message: "{message}"

Is this a reminder request? Reply ONLY with valid JSON — no markdown, no extra text.

If YES:
{{"is_reminder": true, "task": "clean description of what to remind", "remind_at": "YYYY-MM-DD HH:MM"}}

If NO:
{{"is_reminder": false}}

Rules:
- remind_at must be 24-hour format
- Calculate exact datetime for "tomorrow", "in 2 hours", "next Monday", "at 5pm" etc from the current IST time
- remind_at must be in the future"""

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        logger.info("[REMINDER_PARSER] %s", data)
        return data

    except json.JSONDecodeError as exc:
        logger.error("[REMINDER_PARSER] JSON error: %s | raw: %s", exc, raw)
        return None
    except Exception as exc:
        logger.error("[REMINDER_PARSER] Error: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# AI REPLY
# ─────────────────────────────────────────────────────────────────────────────
REMINDER_KEYWORDS = [
    "remind", "reminder", "alert", "notify", "don't forget",
    "याद", "याद दिलाओ", "रिमाइंडर", "set a reminder", "set reminder",
]


def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] from=%s  msg=%s", sender, prompt[:120])

    if not GROQ_API_KEY or groq_client is None:
        return "Sorry, AI is not configured correctly."

    try:
        # ── Reminder path ─────────────────────────────────────────────────────
        if any(kw in prompt.lower() for kw in REMINDER_KEYWORDS):
            logger.info("[AI] Possible reminder request detected")
            data = parse_reminder_with_ai(prompt)

            if data and data.get("is_reminder"):
                task        = data.get("task", prompt)
                time_str    = data.get("remind_at", "")
                try:
                    naive_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    aware_dt = naive_dt.replace(tzinfo=IST)

                    if aware_dt <= datetime.now(IST):
                        return (
                            "⚠️ That time is already in the past!\n"
                            "Please give me a future time, e.g. _'remind me in 2 hours to call John'_"
                        )

                    save_reminder(phone=sender, task=task, remind_at=aware_dt)
                    formatted = aware_dt.strftime("%d %b %Y at %I:%M %p IST")
                    return (
                        f"✅ *Reminder set!*\n"
                        f"📋 Task: {task}\n"
                        f"⏰ Time: {formatted}\n\n"
                        f"I'll message you then!"
                    )
                except ValueError:
                    return (
                        "Sorry, I couldn't understand that time. "
                        "Try: _'Remind me tomorrow at 5pm to call John'_"
                    )

        # ── Normal chat path ──────────────────────────────────────────────────
        history = conversation_history.setdefault(sender, [])
        history.append({"role": "user", "content": prompt})
        recent = history[-10:]

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are a helpful and friendly WhatsApp assistant.\n"
                        "- Keep replies short and clear (under 100 words)\n"
                        "- Be conversational and warm\n"
                        "- Reply in the same language the user uses\n"
                        "- Never make up false information"
                    ),
                },
                *recent,
            ],
            max_tokens=300,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        logger.info("[AI] ✅ reply=%s", reply[:200])
        return reply

    except Exception as exc:
        logger.exception("[AI] Unhandled error: %s", exc)
        return "Sorry, I hit an error. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK — VERIFY
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    p             = dict(request.query_params)
    mode          = p.get("hub.mode")
    verify_token  = p.get("hub.verify_token")
    challenge     = p.get("hub.challenge")

    if verify_token == VERIFY_TOKEN and challenge:
        logger.info("[VERIFY] ✅ Webhook verified")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("[VERIFY] ❌ Mismatch. got token=%s", verify_token)
    raise HTTPException(status_code=403, detail="Token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK — RECEIVE MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(payload: Dict[str, Any]):
    logger.info("[WEBHOOK] Incoming payload")

    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Skip pure status updates
                if "statuses" in value and "messages" not in value:
                    continue

                messages = value.get("messages") or []
                if not messages:
                    continue

                pid = value.get("metadata", {}).get("phone_number_id") or PHONE_NUMBER_ID

                for msg in messages:
                    sender   = msg.get("from")
                    msg_type = msg.get("type")
                    if not sender:
                        continue

                    if msg_type == "text":
                        body  = msg.get("text", {}).get("body", "")
                        logger.info("[MSG] from=%s  body=%s", sender, body)
                        reply = generate_ai_reply(body, sender)
                    elif msg_type == "image":
                        reply = "I received your image! I can only process text for now. 😊"
                    elif msg_type == "audio":
                        reply = "I received your voice message! I can only process text for now. 😊"
                    elif msg_type == "document":
                        reply = "I received your document! I can only process text for now. 😊"
                    else:
                        logger.info("[MSG] Unsupported type '%s' — skipping", msg_type)
                        continue

                    try:
                        send_whatsapp_text(phone_number=sender, message=reply, phone_number_id=pid)
                    except Exception as exc:
                        logger.error("[MSG] ❌ Send failed: %s", exc)

    except Exception as exc:
        logger.exception("[WEBHOOK] ❌ Unhandled: %s", exc)

    return {"status": "received"}


# ─────────────────────────────────────────────────────────────────────────────
# TEST ENDPOINT — send yourself a message to verify token / phone id are correct
# GET /test-send?to=919876543210
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/test-send")
async def test_send(to: str):
    """
    Quick sanity-check. Hit this URL with your phone number (with country code,
    no +) to verify that WHATSAPP_TOKEN and PHONE_NUMBER_ID are working.
    Example: /test-send?to=919876543210
    """
    try:
        result = send_whatsapp_text(
            phone_number=to,
            message="✅ Test message from your WhatsApp bot. Token & Phone ID are working!",
        )
        return {"status": "sent ✅", "meta_response": result}
    except Exception as exc:
        return {"status": "failed ❌", "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    conn    = sqlite3.connect(DB_FILE)
    pending = conn.execute("SELECT COUNT(*) FROM reminders WHERE sent=0").fetchone()[0]
    total   = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
    upcoming = conn.execute(
        "SELECT phone, task, remind_at FROM reminders WHERE sent=0 ORDER BY remind_at LIMIT 5"
    ).fetchall()
    conn.close()

    return {
        "status"            : "ok ✅",
        "current_time_ist"  : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "groq_api_key"      : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "whatsapp_token"    : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "whatsapp_token_prefix": WHATSAPP_TOKEN[:12] + "..." if WHATSAPP_TOKEN else "MISSING",
        "phone_number_id"   : PHONE_NUMBER_ID or "MISSING ❌",
        "groq_sdk"          : "OK ✅" if groq_client else "ERROR ❌",
        "scheduler"         : "RUNNING ✅",
        "reminders_pending" : pending,
        "reminders_total"   : total,
        "upcoming_reminders": [
            {"phone": r[0], "task": r[1], "remind_at": r[2]} for r in upcoming
        ],
        "users_in_memory"   : len(conversation_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)