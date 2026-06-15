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

# ── PostgreSQL ────────────────────────────────────────────────────────────────
import psycopg2
from psycopg2.extras import RealDictCursor

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN      = os.getenv("VERIFY_TOKEN", "my_secret_token_123")
DATABASE_URL          = os.getenv("DATABASE_URL", "")
TWILIO_ACCOUNT_SID    = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN     = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER   = os.getenv("TWILIO_PHONE_NUMBER", "")
GROQ_MODEL            = "llama-3.3-70b-versatile"
GRAPH_API_VERSION     = "v22.0"

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

# ── Twilio ────────────────────────────────────────────────────────────────────
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
    logger.info("Twilio   : OK ✅")
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio NOT installed — calls disabled. Run: pip install twilio")

# ── Conversation Memory ───────────────────────────────────────────────────────
conversation_history: Dict[str, list] = {}

app = FastAPI(title="WhatsApp AI Bot with Reminders")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE — PostgreSQL (persistent, survives Railway restarts)
# ─────────────────────────────────────────────────────────────────────────────
def get_conn():
    """Get a fresh PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id         SERIAL PRIMARY KEY,
                    phone      TEXT NOT NULL,
                    task       TEXT NOT NULL,
                    remind_at  TIMESTAMP NOT NULL,
                    recurrence TEXT DEFAULT 'none',
                    sent       BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute("""
                ALTER TABLE reminders
                ADD COLUMN IF NOT EXISTS recurrence TEXT DEFAULT 'none'
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS contacts (
                    id         SERIAL PRIMARY KEY,
                    owner      TEXT NOT NULL,
                    phone      TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(owner, phone)
                )
            """)
        conn.commit()
    logger.info("Database initialised ✅ (PostgreSQL)")


def save_reminder(phone: str, task: str, remind_at: datetime, recurrence: str = "none"):
    """Store reminder. remind_at may be naive (assumed IST) or aware."""
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=IST)
    else:
        remind_at = remind_at.astimezone(IST)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reminders (phone, task, remind_at, recurrence) VALUES (%s, %s, %s, %s)",
                (phone, task, remind_at, recurrence),
            )
        conn.commit()
    logger.info("[DB] Saved reminder: '%s' at %s recurrence=%s for %s", task, remind_at, recurrence, phone)


def get_due_reminders():
    """Return reminders whose remind_at <= now (IST), not yet sent."""
    now = datetime.now(IST)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, phone, task, remind_at, recurrence FROM reminders "
                "WHERE remind_at <= %s AND sent = FALSE ORDER BY remind_at",
                (now,),
            )
            rows = cur.fetchall()
    return rows


def mark_sent(reminder_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE reminders SET sent = TRUE WHERE id = %s", (reminder_id,))
        conn.commit()


def reschedule_reminder(reminder_id: int, current_time: datetime, recurrence: str):
    """
    For recurring reminders — calculate next fire time and reset sent=FALSE.
    recurrence: hourly | daily | weekly | monthly
    """
    from datetime import timedelta

    if recurrence == "hourly":
        next_time = current_time + timedelta(hours=1)
    elif recurrence == "daily":
        next_time = current_time + timedelta(days=1)
    elif recurrence == "weekly":
        next_time = current_time + timedelta(weeks=1)
    elif recurrence == "monthly":
        # Same day next month
        month = current_time.month + 1 if current_time.month < 12 else 1
        year  = current_time.year + 1 if current_time.month == 12 else current_time.year
        try:
            next_time = current_time.replace(year=year, month=month)
        except ValueError:
            # Handle months with fewer days (e.g. Feb 30)
            next_time = current_time.replace(year=year, month=month, day=28)
    else:
        return  # not recurring

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reminders SET remind_at = %s, sent = FALSE WHERE id = %s",
                (next_time, reminder_id),
            )
        conn.commit()
    logger.info("[DB] Rescheduled #%d → next: %s (%s)", reminder_id, next_time, recurrence)


# ─────────────────────────────────────────────────────────────────────────────
# CONTACTS DB FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_contact(owner: str, phone: str, name: str) -> bool:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO contacts (owner, phone, name) VALUES (%s, %s, %s) "
                    "ON CONFLICT (owner, phone) DO UPDATE SET name = %s",
                    (owner, phone, name, name),
                )
            conn.commit()
        logger.info("[CONTACTS] Saved %s (%s) for %s", name, phone, owner)
        return True
    except Exception as exc:
        logger.error("[CONTACTS] Add error: %s", exc)
        return False


def get_contacts(owner: str) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT phone, name FROM contacts WHERE owner = %s ORDER BY name",
                (owner,),
            )
            return cur.fetchall()


def remove_contact(owner: str, phone: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM contacts WHERE owner = %s AND phone = %s",
                (owner, phone),
            )
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted


def get_pending_reminders_for_user(phone: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, task, remind_at, recurrence FROM reminders "
                "WHERE phone = %s AND sent = FALSE ORDER BY remind_at",
                (phone,),
            )
            return cur.fetchall()


def cancel_reminder_for_user(phone: str, reminder_id: int) -> bool:
    """Cancel a specific reminder for a user. Returns True if deleted."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM reminders WHERE id = %s AND phone = %s AND sent = FALSE",
                (reminder_id, phone),
            )
            deleted = cur.rowcount > 0
        conn.commit()
    return deleted
    """Return (pending, total) counts for health check."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM reminders WHERE sent = FALSE")
            pending = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reminders")
            total = cur.fetchone()[0]
    return pending, total


def get_upcoming_reminders(limit: int = 5):
    """Return next N pending reminders for health check."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT phone, task, remind_at FROM reminders "
                "WHERE sent = FALSE ORDER BY remind_at LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    return rows


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


def make_twilio_call(phone: str, task: str):
    """
    Make an automated voice call using Twilio.
    Phone number must include country code e.g. +919876543210
    """
    if not TWILIO_AVAILABLE:
        logger.warning("[CALL] Twilio not installed — skipping call")
        return

    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN or not TWILIO_PHONE_NUMBER:
        logger.warning("[CALL] Twilio credentials missing — skipping call")
        return

    try:
        # Format phone — WhatsApp numbers come as 919876543210, need +919876543210
        to_number = phone if phone.startswith("+") else f"+{phone}"

        client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # TwiML — what Twilio says when call is picked up
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="alice" language="en-IN">
        Hello! This is your WhatsApp bot reminder.
        {task}.
        I repeat, {task}.
        Have a great day!
    </Say>
    <Pause length="1"/>
</Response>"""

        call = client.calls.create(
            twiml=twiml,
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
        )
        logger.info("[CALL] ✅ Call initiated to %s — SID: %s", to_number, call.sid)

    except Exception as exc:
        logger.error("[CALL] ❌ Call failed: %s", exc)


def send_reminder_with_voice(phone: str, task: str):
    """
    Send reminder as text + WhatsApp voice note + phone call.
    Each step is independent — if one fails, others still go through.
    """
    # 1. Always send text first
    try:
        send_whatsapp_text(
            phone_number=phone,
            message=f"⏰ *Reminder:* {task}",
        )
        logger.info("[REMINDER] ✅ Text sent to %s", phone)
    except Exception as exc:
        logger.error("[REMINDER] ❌ Text failed: %s", exc)

    # 2. Send WhatsApp voice note
    voice_text = f"Hey! This is your reminder. {task}"
    audio_path = generate_voice_note(voice_text)
    if audio_path:
        try:
            send_whatsapp_audio(phone_number=phone, audio_path=audio_path)
            logger.info("[REMINDER] ✅ Voice note sent to %s", phone)
        except Exception as exc:
            logger.error("[REMINDER] ❌ Voice note failed: %s", exc)

    # 3. Make phone call via Twilio
    try:
        make_twilio_call(phone=phone, task=task)
    except Exception as exc:
        logger.error("[REMINDER] ❌ Call failed: %s", exc)


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

            for reminder_id, phone, task, remind_at, recurrence in due:
                logger.info("[SCHEDULER] Firing reminder #%d → %s : %s (recurrence=%s)", reminder_id, phone, task, recurrence)
                try:
                    send_reminder_with_voice(phone=phone, task=task)

                    if recurrence and recurrence != "none":
                        # Recurring — reschedule for next occurrence
                        reschedule_reminder(reminder_id, remind_at, recurrence)
                        logger.info("[SCHEDULER] ✅ Rescheduled recurring #%d (%s)", reminder_id, recurrence)
                    else:
                        # One-time — mark done
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

If YES (one-time):
{{"is_reminder": true, "task": "clean description", "remind_at": "YYYY-MM-DD HH:MM", "recurrence": "none"}}

If YES (recurring):
{{"is_reminder": true, "task": "clean description", "remind_at": "YYYY-MM-DD HH:MM", "recurrence": "hourly|daily|weekly|monthly"}}

If NO:
{{"is_reminder": false}}

Rules:
- remind_at must be 24-hour format and in the future
- recurrence = "daily" for "every day", "weekly" for "every week/Monday", "monthly" for "every month", "hourly" for "every hour"
- recurrence = "none" for one-time reminders
- Calculate exact first occurrence from current IST time"""

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
# WHISPER — transcribe voice messages using Groq's free Whisper API
# ─────────────────────────────────────────────────────────────────────────────
def download_whatsapp_audio(media_id: str) -> str | None:
    """
    Download audio file from WhatsApp media API.
    Returns temp file path or None on failure.
    """
    try:
        # Step 1 — get download URL from media ID
        url  = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            timeout=15,
        )
        if not resp.ok:
            logger.error("[WHISPER] Failed to get media URL: %s", resp.text)
            return None

        download_url = resp.json().get("url")
        if not download_url:
            logger.error("[WHISPER] No URL in media response")
            return None

        # Step 2 — download the audio file
        audio_resp = requests.get(
            download_url,
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            timeout=30,
        )
        if not audio_resp.ok:
            logger.error("[WHISPER] Failed to download audio: %s", audio_resp.status_code)
            return None

        # Save to temp file as .ogg (WhatsApp audio format)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
        tmp.write(audio_resp.content)
        tmp.close()
        logger.info("[WHISPER] ✅ Audio downloaded: %s (%d bytes)", tmp.name, len(audio_resp.content))
        return tmp.name

    except Exception as exc:
        logger.error("[WHISPER] Download error: %s", exc)
        return None


def transcribe_audio(audio_path: str) -> str | None:
    """
    Transcribe audio file using Groq's free Whisper API.
    Returns transcribed text or None on failure.
    """
    try:
        with open(audio_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )
        # Clean up temp file
        try:
            os.remove(audio_path)
        except Exception:
            pass

        text = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
        logger.info("[WHISPER] ✅ Transcribed: %s", text[:200])
        return text

    except Exception as exc:
        logger.error("[WHISPER] Transcription error: %s", exc)
        return None


def handle_voice_message(msg: dict, sender: str) -> str:
    """
    Full pipeline: download → transcribe → AI reply.
    """
    try:
        # Get media ID from message
        audio_data = msg.get("audio") or msg.get("voice") or {}
        media_id   = audio_data.get("id")

        if not media_id:
            return "I received your voice message but couldn't process it. Please try again! 😊"

        logger.info("[WHISPER] Processing voice message from %s media_id=%s", sender, media_id)

        # Download audio
        audio_path = download_whatsapp_audio(media_id)
        if not audio_path:
            return "Sorry, I couldn't download your voice message. Please try again! 🎤"

        # Transcribe
        text = transcribe_audio(audio_path)
        if not text:
            return "Sorry, I couldn't understand your voice message. Please speak clearly or type instead! 🎤"

        logger.info("[WHISPER] ✅ Voice message from %s: %s", sender, text[:100])

        # Reply with transcription + AI response
        ai_reply = generate_ai_reply(text, sender)
        return f"🎤 _I heard: \"{text}\"_\n\n{ai_reply}"

    except Exception as exc:
        logger.error("[WHISPER] handle_voice_message error: %s", exc)
        return "Sorry, I had trouble processing your voice message. Please try again! 🎤"



REMINDER_KEYWORDS = [
    "remind", "reminder", "alert", "notify", "don't forget",
    "याद", "याद दिलाओ", "रिमाइंडर", "set a reminder", "set reminder",
]

LIST_KEYWORDS   = ["my reminders", "show reminders", "list reminders",
                   "my reminder", "reminders list", "what are my reminders",
                   "मेरे रिमाइंडर"]

CANCEL_KEYWORDS = ["cancel reminder", "delete reminder", "remove reminder",
                   "cancel reminders", "रिमाइंडर कैंसिल"]

BROADCAST_KEYWORDS  = ["broadcast:", "broadcast :", "send to all", "announce:"]
ADD_CONTACT_KEYWORDS    = ["add contact", "save contact", "new contact"]
LIST_CONTACT_KEYWORDS   = ["my contacts", "show contacts", "list contacts", "contact list"]
REMOVE_CONTACT_KEYWORDS = ["remove contact", "delete contact"]


def handle_add_contact(sender: str, prompt: str) -> str:
    """Parse 'add contact 919876543210 John' and save."""
    import re
    # Extract phone number (10+ digits)
    phone_match = re.search(r'\d{10,15}', prompt)
    if not phone_match:
        return (
            "Please include the phone number with country code.\n"
            "Example: _'add contact 919876543210 John'_"
        )
    phone = phone_match.group()
    # Extract name — everything after the phone number
    name_part = prompt[prompt.index(phone) + len(phone):].strip()
    name = name_part if name_part else phone

    add_contact(owner=sender, phone=phone, name=name)
    return (
        f"✅ Contact saved!\n"
        f"👤 Name: {name}\n"
        f"📞 Phone: +{phone}\n\n"
        f"Type *my contacts* to see all contacts.\n"
        f"To broadcast: _'broadcast: Your message here'_"
    )


def handle_list_contacts(sender: str) -> str:
    contacts = get_contacts(sender)
    if not contacts:
        return (
            "📭 No contacts saved yet!\n\n"
            "Add one: _'add contact 919876543210 John'_"
        )
    lines = [f"👥 *Your Contacts ({len(contacts)}):*\n"]
    for phone, name in contacts:
        lines.append(f"👤 {name} — +{phone}")
    lines.append("\n_To remove: 'remove contact 919876543210'_")
    lines.append("_To broadcast: 'broadcast: Your message'_")
    return "\n".join(lines)


def handle_remove_contact(sender: str, prompt: str) -> str:
    import re
    phone_match = re.search(r'\d{10,15}', prompt)
    if not phone_match:
        return "Please include the phone number.\nExample: _'remove contact 919876543210'_"
    phone   = phone_match.group()
    deleted = remove_contact(owner=sender, phone=phone)
    if deleted:
        return f"✅ Contact +{phone} removed!\n\nType *my contacts* to see remaining contacts."
    return f"❌ Contact +{phone} not found in your list.\nType *my contacts* to see saved contacts."


def handle_broadcast(sender: str, prompt: str) -> str:
    """Extract message after 'broadcast:' and send to all contacts."""
    # Extract message after broadcast keyword
    message = ""
    for kw in BROADCAST_KEYWORDS:
        if kw in prompt.lower():
            idx     = prompt.lower().index(kw)
            message = prompt[idx + len(kw):].strip()
            break

    if not message:
        return (
            "Please include a message after 'broadcast:'.\n"
            "Example: _'broadcast: Team meeting at 10am tomorrow'_"
        )

    contacts = get_contacts(sender)
    if not contacts:
        return (
            "📭 No contacts to broadcast to!\n\n"
            "First add contacts: _'add contact 919876543210 John'_"
        )

    success_count = 0
    fail_count    = 0

    for phone, name in contacts:
        try:
            send_whatsapp_text(
                phone_number=phone,
                message=f"📢 *Announcement:*\n{message}",
            )
            success_count += 1
            logger.info("[BROADCAST] ✅ Sent to %s (%s)", name, phone)
            time.sleep(0.5)  # small delay to avoid rate limiting
        except Exception as exc:
            fail_count += 1
            logger.error("[BROADCAST] ❌ Failed to %s (%s): %s", name, phone, exc)

    result = f"📢 *Broadcast Complete!*\n\n"
    result += f"✅ Sent to: {success_count} contact(s)\n"
    if fail_count:
        result += f"❌ Failed: {fail_count} contact(s)\n"
    result += f"\n📝 Message: _{message}_"
    return result


def handle_list_reminders(sender: str) -> str:
    rows = get_pending_reminders_for_user(sender)
    if not rows:
        return "📭 You have no pending reminders!\n\nSend me something like:\n_'Remind me tomorrow at 5pm to call John'_"

    lines = ["📋 *Your Pending Reminders:*\n"]
    for rid, task, remind_at, recurrence in rows:
        time_str   = remind_at.strftime("%d %b %Y at %I:%M %p IST") if remind_at else "—"
        recur_str  = f" 🔁 {recurrence}" if recurrence and recurrence != "none" else ""
        lines.append(f"*#{rid}* — {task}\n⏰ {time_str}{recur_str}")

    lines.append("\n_To cancel: type 'cancel reminder 2' (use the # number)_")
    return "\n\n".join(lines)


def handle_cancel_reminder(sender: str, prompt: str) -> str:
    import re
    # Extract number from message e.g. "cancel reminder 3" → 3
    match = re.search(r'\d+', prompt)
    if not match:
        return (
            "Please tell me which reminder to cancel.\n"
            "Example: _'cancel reminder 2'_\n\n"
            "Type *my reminders* to see the list with IDs."
        )

    reminder_id = int(match.group())
    deleted     = cancel_reminder_for_user(sender, reminder_id)

    if deleted:
        logger.info("[CANCEL] ✅ Reminder #%d cancelled for %s", reminder_id, sender)
        return f"✅ Reminder *#{reminder_id}* has been cancelled!\n\nType *my reminders* to see remaining reminders."
    else:
        return (
            f"❌ Couldn't find reminder *#{reminder_id}*.\n"
            "It may not exist or already completed.\n\n"
            "Type *my reminders* to see your active reminders."
        )


def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] from=%s  msg=%s", sender, prompt[:120])

    if not GROQ_API_KEY or groq_client is None:
        return "Sorry, AI is not configured correctly."

    try:
        p_lower  = prompt.lower().strip()

        # ── STRICT COMMAND DETECTION (checked before AI) ──────────────────────
        # These are checked by startswith so AI never intercepts them

        if p_lower.startswith("add contact"):
            logger.info("[AI] Add contact command from %s", sender)
            return handle_add_contact(sender, prompt)

        if p_lower.startswith("remove contact") or p_lower.startswith("delete contact"):
            logger.info("[AI] Remove contact command from %s", sender)
            return handle_remove_contact(sender, prompt)

        if p_lower.startswith("my contacts") or p_lower.startswith("show contacts") or p_lower.startswith("list contacts"):
            logger.info("[AI] List contacts command from %s", sender)
            return handle_list_contacts(sender)

        if p_lower.startswith("broadcast:") or p_lower.startswith("broadcast :"):
            logger.info("[AI] Broadcast command from %s", sender)
            return handle_broadcast(sender, prompt)

        if p_lower.startswith("my reminders") or p_lower.startswith("show reminders") or p_lower.startswith("list reminders"):
            logger.info("[AI] List reminders command from %s", sender)
            return handle_list_reminders(sender)

        if p_lower.startswith("cancel reminder") or p_lower.startswith("delete reminder"):
            logger.info("[AI] Cancel reminder command from %s", sender)
            return handle_cancel_reminder(sender, prompt)

        # ── FUZZY KEYWORD DETECTION (for natural language) ────────────────────
        # ── List reminders ────────────────────────────────────────────────────
        if any(kw in p_lower for kw in LIST_KEYWORDS):
            logger.info("[AI] List reminders request from %s", sender)
            return handle_list_reminders(sender)

        # ── Cancel reminder ───────────────────────────────────────────────────
        if any(kw in p_lower for kw in CANCEL_KEYWORDS):
            logger.info("[AI] Cancel reminder request from %s", sender)
            return handle_cancel_reminder(sender, prompt)

        # ── Add contact ───────────────────────────────────────────────────────
        if any(kw in p_lower for kw in ADD_CONTACT_KEYWORDS):
            logger.info("[AI] Add contact request from %s", sender)
            return handle_add_contact(sender, prompt)

        # ── List contacts ─────────────────────────────────────────────────────
        if any(kw in p_lower for kw in LIST_CONTACT_KEYWORDS):
            logger.info("[AI] List contacts request from %s", sender)
            return handle_list_contacts(sender)

        # ── Remove contact ────────────────────────────────────────────────────
        if any(kw in p_lower for kw in REMOVE_CONTACT_KEYWORDS):
            logger.info("[AI] Remove contact request from %s", sender)
            return handle_remove_contact(sender, prompt)

        # ── Broadcast ─────────────────────────────────────────────────────────
        if any(kw in p_lower for kw in BROADCAST_KEYWORDS):
            logger.info("[AI] Broadcast request from %s", sender)
            return handle_broadcast(sender, prompt)

        # ── Reminder path ─────────────────────────────────────────────────────
        if any(kw in p_lower for kw in REMINDER_KEYWORDS):
            logger.info("[AI] Possible reminder request detected")
            data = parse_reminder_with_ai(prompt)

            if data and data.get("is_reminder"):
                task        = data.get("task", prompt)
                time_str    = data.get("remind_at", "")
                recurrence  = data.get("recurrence", "none")
                try:
                    naive_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    aware_dt = naive_dt.replace(tzinfo=IST)

                    if aware_dt <= datetime.now(IST):
                        return (
                            "⚠️ That time is already in the past!\n"
                            "Please give me a future time, e.g. _'remind me in 2 hours to call John'_"
                        )

                    save_reminder(phone=sender, task=task, remind_at=aware_dt, recurrence=recurrence)
                    formatted = aware_dt.strftime("%d %b %Y at %I:%M %p IST")

                    if recurrence and recurrence != "none":
                        return (
                            f"✅ *Recurring Reminder set!*\n"
                            f"📋 Task: {task}\n"
                            f"⏰ First reminder: {formatted}\n"
                            f"🔁 Repeats: {recurrence.capitalize()}\n\n"
                            f"I'll keep reminding you!"
                        )
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
                    elif msg_type == "audio":
                        logger.info("[MSG] Voice message from %s — transcribing...", sender)
                        reply = handle_voice_message(msg, sender)
                    elif msg_type == "image":
                        reply = "I received your image! I can only process text and voice for now. 😊"
                    elif msg_type == "document":
                        reply = "I received your document! I can only process text and voice for now. 😊"
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
    try:
        pending, total = get_reminder_counts()
        upcoming       = get_upcoming_reminders()
        db_status      = "OK ✅"
    except Exception as exc:
        pending, total, upcoming = 0, 0, []
        db_status = f"ERROR ❌ {exc}"

    return {
        "status"            : "ok ✅",
        "current_time_ist"  : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "database"          : db_status,
        "groq_api_key"      : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "whatsapp_token"    : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "whatsapp_token_prefix": WHATSAPP_TOKEN[:12] + "..." if WHATSAPP_TOKEN else "MISSING",
        "phone_number_id"   : PHONE_NUMBER_ID or "MISSING ❌",
        "groq_sdk"          : "OK ✅" if groq_client else "ERROR ❌",
        "gtts"              : "OK ✅" if GTTS_AVAILABLE else "NOT INSTALLED ⚠️",
        "twilio"            : "OK ✅" if (TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID) else "NOT CONFIGURED ⚠️",
        "scheduler"         : "RUNNING ✅",
        "reminders_pending" : pending,
        "reminders_total"   : total,
        "upcoming_reminders": [
            {"phone": r[0], "task": r[1], "remind_at": str(r[2])} for r in upcoming
        ],
        "users_in_memory"   : len(conversation_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOBILE APP API ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/reminders")
async def api_get_reminders():
    """Get all reminders for mobile app."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, phone, task, remind_at, recurrence, sent
                FROM reminders ORDER BY sent ASC, remind_at ASC
            """)
            rows = cur.fetchall()
    return [
        {
            "id": r[0], "phone": r[1], "task": r[2],
            "remind_at": str(r[3]), "recurrence": r[4] or "none",
            "sent": r[5], "call_status": "Completed" if r[5] else "Pending",
        }
        for r in rows
    ]


@app.delete("/api/reminders/{reminder_id}")
async def api_delete_reminder(reminder_id: int):
    """Cancel a reminder from mobile app."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM reminders WHERE id = %s", (reminder_id,))
            deleted = cur.rowcount > 0
        conn.commit()
    if not deleted:
        raise HTTPException(status_code=404, detail="Reminder not found")
    return {"status": "deleted"}


@app.patch("/api/reminders/{reminder_id}")
async def api_update_reminder(reminder_id: int, body: Dict[str, Any]):
    """Edit a reminder from mobile app."""
    task      = body.get("task")
    remind_at = body.get("remind_at")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE reminders SET task = %s, remind_at = %s WHERE id = %s",
                (task, remind_at, reminder_id),
            )
        conn.commit()
    return {"status": "updated"}


@app.get("/api/contacts/{owner}")
async def api_get_contacts(owner: str):
    """Get contacts for mobile app."""
    rows = get_contacts(owner)
    return [{"phone": r[0], "name": r[1]} for r in rows]


@app.post("/api/contacts")
async def api_add_contact(body: Dict[str, Any]):
    """Add contact from mobile app."""
    add_contact(
        owner=body.get("owner"),
        phone=body.get("phone"),
        name=body.get("name"),
    )
    return {"status": "saved"}


@app.delete("/api/contacts/{owner}/{phone}")
async def api_delete_contact(owner: str, phone: str):
    """Remove contact from mobile app."""
    deleted = remove_contact(owner=owner, phone=phone)
    if not deleted:
        raise HTTPException(status_code=404, detail="Contact not found")
    return {"status": "deleted"}


@app.post("/api/broadcast")
async def api_broadcast(body: Dict[str, Any]):
    """Broadcast from mobile app to selected contacts."""
    owner   = body.get("owner")
    message = body.get("message")
    phones  = body.get("phones", [])

    if not message or not phones:
        raise HTTPException(status_code=400, detail="message and phones required")

    success = 0
    failed  = 0
    for phone in phones:
        try:
            send_whatsapp_text(phone_number=phone, message=f"📢 *Announcement:*\n{message}")
            success += 1
            time.sleep(0.5)
        except Exception as exc:
            logger.error("[API_BROADCAST] Failed to %s: %s", phone, exc)
            failed += 1

    return {"status": "done", "success": success, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/dashboard")
async def dashboard():
    from fastapi.responses import HTMLResponse

    try:
        now_ist = datetime.now(IST).strftime("%d %b %Y %I:%M:%S %p IST")

        # Fetch all reminders and contacts
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, phone, task, remind_at, recurrence, sent, created_at
                    FROM reminders ORDER BY sent ASC, remind_at ASC
                """)
                all_reminders = cur.fetchall()
                cur.execute("SELECT COUNT(*) FROM contacts")
                total_contacts = cur.fetchone()[0]

        pending   = [r for r in all_reminders if not r[5]]
        completed = [r for r in all_reminders if r[5]]

        def reminder_rows(rows):
            if not rows:
                return "<tr><td colspan='6' style='text-align:center;color:#888'>No records</td></tr>"
            html = ""
            for r in rows:
                rid, phone, task, remind_at, recurrence, sent, created_at = r
                recurrence  = recurrence or "none"
                badge_color = "#28a745" if recurrence != "none" else "#6c757d"
                time_str    = remind_at.strftime("%d %b %Y %I:%M %p") if remind_at else "—"
                html += f"""
                <tr>
                    <td>{rid}</td>
                    <td>+{phone}</td>
                    <td>{task}</td>
                    <td>{time_str}</td>
                    <td><span style="background:{badge_color};color:white;padding:2px 8px;border-radius:10px;font-size:12px">{recurrence}</span></td>
                    <td><span style="background:{'#ffc107' if not sent else '#28a745'};color:white;padding:2px 8px;border-radius:10px;font-size:12px">{'pending' if not sent else 'done'}</span></td>
                </tr>"""
            return html

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>WhatsApp Bot Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #333; }}
        .header {{ background: linear-gradient(135deg, #25D366, #128C7E); color: white; padding: 24px 32px; }}
        .header h1 {{ font-size: 26px; }}
        .header p {{ opacity: 0.85; margin-top: 4px; font-size: 14px; }}
        .container {{ max-width: 1100px; margin: 24px auto; padding: 0 16px; }}
        .cards {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
        .card {{ background: white; border-radius: 12px; padding: 20px 24px; flex: 1; min-width: 180px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.07); text-align: center; }}
        .card .num {{ font-size: 36px; font-weight: bold; color: #128C7E; }}
        .card .label {{ color: #666; margin-top: 4px; font-size: 14px; }}
        .status-bar {{ background: white; border-radius: 12px; padding: 16px 24px; margin-bottom: 24px;
                       box-shadow: 0 2px 8px rgba(0,0,0,0.07); display: flex; gap: 24px; flex-wrap: wrap; }}
        .status-item {{ font-size: 14px; }}
        .status-item span {{ font-weight: 600; }}
        .section {{ background: white; border-radius: 12px; padding: 20px 24px; margin-bottom: 24px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.07); }}
        .section h2 {{ font-size: 18px; margin-bottom: 16px; color: #128C7E; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
        th {{ background: #f8f9fa; padding: 10px 12px; text-align: left; font-weight: 600; color: #555;
              border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #f0f0f0; }}
        tr:hover td {{ background: #f8fffe; }}
        .refresh {{ color: #888; font-size: 12px; text-align: center; margin-top: 16px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 WhatsApp Bot Dashboard</h1>
        <p>Live status — auto refreshes every 30 seconds &nbsp;|&nbsp; {now_ist}</p>
    </div>
    <div class="container">

        <!-- Stats Cards -->
        <div class="cards">
            <div class="card">
                <div class="num">{len(pending)}</div>
                <div class="label">⏳ Pending Reminders</div>
            </div>
            <div class="card">
                <div class="num">{len(completed)}</div>
                <div class="label">✅ Completed</div>
            </div>
            <div class="card">
                <div class="num">{len([r for r in pending if r[4] and r[4] != 'none'])}</div>
                <div class="label">🔁 Recurring Active</div>
            </div>
            <div class="card">
                <div class="num">{total_contacts}</div>
                <div class="label">👥 Broadcast Contacts</div>
            </div>
        </div>

        <!-- System Status -->
        <div class="status-bar">
            <div class="status-item">🤖 Bot: <span style="color:#28a745">RUNNING</span></div>
            <div class="status-item">🧠 AI: <span style="color:#28a745">{'ON' if groq_client else 'OFF'}</span></div>
            <div class="status-item">🗄️ DB: <span style="color:#28a745">PostgreSQL</span></div>
            <div class="status-item">🎙️ Voice: <span style="color:{'#28a745' if GTTS_AVAILABLE else '#dc3545'}">{'ON' if GTTS_AVAILABLE else 'OFF'}</span></div>
            <div class="status-item">📞 Calls: <span style="color:{'#28a745' if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID else '#dc3545'}">{'ON' if TWILIO_AVAILABLE and TWILIO_ACCOUNT_SID else 'OFF'}</span></div>
        </div>

        <!-- Pending Reminders -->
        <div class="section">
            <h2>⏳ Pending Reminders ({len(pending)})</h2>
            <table>
                <tr>
                    <th>ID</th><th>Phone</th><th>Task</th><th>Scheduled Time</th><th>Recurrence</th><th>Status</th>
                </tr>
                {reminder_rows(pending)}
            </table>
        </div>

        <!-- Completed Reminders -->
        <div class="section">
            <h2>✅ Completed Reminders ({len(completed)})</h2>
            <table>
                <tr>
                    <th>ID</th><th>Phone</th><th>Task</th><th>Was Scheduled</th><th>Recurrence</th><th>Status</th>
                </tr>
                {reminder_rows(completed)}
            </table>
        </div>

        <div class="refresh">🔄 Page auto-refreshes every 30 seconds</div>
    </div>
</body>
</html>"""
        return HTMLResponse(content=html)

    except Exception as exc:
        logger.exception("[DASHBOARD] Error: %s", exc)
        return HTMLResponse(content=f"<h2>Dashboard error: {exc}</h2>", status_code=500)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
