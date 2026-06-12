"""
WhatsApp AI Bot - FastAPI + Groq + Reminders + Meta WhatsApp Cloud API

.env file:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
WHATSAPP_TOKEN=EAAxxxxxxxxxxxxxxxxxxxxx
PHONE_NUMBER_ID=1200795349780072
VERIFY_TOKEN=my_secret_token_123
RENDER_EXTERNAL_URL=https://whatsapp-ai-bot-faia.onrender.com
"""

import os
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse
import requests

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN      = os.getenv("VERIFY_TOKEN", "my_secret_token_123")
RENDER_URL        = os.getenv("RENDER_EXTERNAL_URL", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GRAPH_API_VERSION = "v25.0"
IST               = ZoneInfo("Asia/Kolkata")

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
logger.info("  PHONE_NUMBER_ID : %s", PHONE_NUMBER_ID or "MISSING ❌")
logger.info("  WHATSAPP_TOKEN  : %s", "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌")
logger.info("  GROQ_API_KEY    : %s", "SET ✅" if GROQ_API_KEY else "MISSING ❌")
logger.info("  GROQ_MODEL      : %s", GROQ_MODEL)
logger.info("  CURRENT IST     : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
logger.info("  RENDER_URL      : %s", RENDER_URL or "NOT SET")
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
    logger.info("[DB] Saved: '%s' at %s for %s", task, remind_at_str, phone)

def get_due_reminders():
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
# SEND WHATSAPP MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp_text(phone_number: str, message: str, phone_number_id: str = "") -> dict:
    pid   = phone_number_id or PHONE_NUMBER_ID
    token = WHATSAPP_TOKEN

    if not token:
        raise ValueError("WHATSAPP_TOKEN is empty — check Render environment variables")
    if not pid:
        raise ValueError("PHONE_NUMBER_ID is empty — check Render environment variables")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to":    phone_number,
        "type":  "text",
        "text":  {"preview_url": False, "body": message},
    }

    logger.info("[SEND] To=%s", phone_number)

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}") from e

    logger.info("[SEND] Status=%s body=%s", resp.status_code, resp.text[:200])

    if resp.status_code == 401:
        raise RuntimeError(
            "401 Token expired! Go to developers.facebook.com → "
            "WhatsApp → API Setup → Generate new token → Update on Render"
        )
    if not resp.ok:
        raise RuntimeError(f"Meta API {resp.status_code}: {resp.text}")

    logger.info("[SEND] ✅ Sent to %s", phone_number)
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# REMINDER SCHEDULER — checks every 30 seconds
# ─────────────────────────────────────────────────────────────────────────────
def reminder_scheduler():
    logger.info("[SCHEDULER] Started ✅")
    while True:
        try:
            due = get_due_reminders()
            if due:
                logger.info("[SCHEDULER] %d due reminder(s)", len(due))
            for reminder_id, phone, task, remind_at in due:
                try:
                    send_whatsapp_text(
                        phone_number=phone,
                        message=f"⏰ *Reminder:* {task}",
                    )
                    mark_sent(reminder_id)
                    logger.info("[SCHEDULER] ✅ Sent #%d: %s", reminder_id, task)
                except Exception as e:
                    logger.error("[SCHEDULER] ❌ Failed #%d: %s", reminder_id, e)
        except Exception as e:
            logger.error("[SCHEDULER] ❌ Loop error: %s", e)
        time.sleep(30)

threading.Thread(target=reminder_scheduler, daemon=True).start()
logger.info("[SCHEDULER] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# KEEP ALIVE — pings /health every 14 min so Render never sleeps
# ─────────────────────────────────────────────────────────────────────────────
def keep_alive():
    time.sleep(60)
    while True:
        if RENDER_URL:
            try:
                resp = requests.get(f"{RENDER_URL}/health", timeout=10)
                logger.info("[KEEPALIVE] ✅ Pinged → %s", resp.status_code)
            except Exception as e:
                logger.warning("[KEEPALIVE] ⚠️ Failed: %s", e)
        time.sleep(840)  # 14 minutes

threading.Thread(target=keep_alive, daemon=True).start()
logger.info("[KEEPALIVE] Started ✅")


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
- Calculate exact datetime for "tomorrow", "in 2 hours", "next Monday", "at 5pm" etc
- remind_at must be in the future
- Use IST timezone"""

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.1,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        logger.info("[PARSER] %s", data)
        return data

    except Exception as e:
        logger.error("[PARSER] Error: %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE AI REPLY
# ─────────────────────────────────────────────────────────────────────────────
REMINDER_KEYWORDS = [
    "remind", "reminder", "alert", "notify", "don't forget",
    "याद", "याद दिलाओ", "रिमाइंडर", "set a reminder", "set reminder",
]

def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] from=%s msg=%s", sender, prompt[:120])

    if not GROQ_API_KEY or groq_client is None:
        return "Sorry, AI is not configured correctly."

    try:
        # ── Reminder check ────────────────────────────────────────────────────
        if any(kw in prompt.lower() for kw in REMINDER_KEYWORDS):
            logger.info("[AI] Reminder request detected")
            data = parse_reminder_with_ai(prompt)

            if data and data.get("is_reminder"):
                task     = data.get("task", prompt)
                time_str = data.get("remind_at", "")
                try:
                    naive_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    aware_dt = naive_dt.replace(tzinfo=IST)

                    if aware_dt <= datetime.now(IST):
                        return (
                            "⚠️ That time is already in the past!\n"
                            "Please give me a future time.\n"
                            "Example: _'Remind me in 2 hours to call John'_"
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
                        "Sorry, I couldn't understand that time.\n"
                        "Try: _'Remind me tomorrow at 5pm to call John'_"
                    )

        # ── Normal AI reply ───────────────────────────────────────────────────
        history = conversation_history.setdefault(sender, [])
        history.append({"role": "user", "content": prompt})
        recent = history[-10:]

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
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
        logger.info("[AI] ✅ %s", reply[:200])
        return reply

    except Exception as e:
        logger.exception("[AI] Error: %s", e)
        return "Sorry, I encountered an error. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    p            = dict(request.query_params)
    verify_token = p.get("hub.verify_token")
    challenge    = p.get("hub.challenge")

    if verify_token == VERIFY_TOKEN and challenge:
        logger.info("[VERIFY] ✅ Verified")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("[VERIFY] ❌ Token mismatch")
    raise HTTPException(status_code=403, detail="Token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# RECEIVE MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(payload: Dict[str, Any]):
    logger.info("[WEBHOOK] Incoming")

    try:
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

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
                        reply = generate_ai_reply(body, sender)
                    elif msg_type == "image":
                        reply = "I received your image! I can only process text for now. 😊"
                    elif msg_type == "audio":
                        reply = "I received your voice message! I can only process text for now. 😊"
                    elif msg_type == "document":
                        reply = "I received your document! I can only process text for now. 😊"
                    else:
                        continue

                    try:
                        send_whatsapp_text(phone_number=sender, message=reply, phone_number_id=pid)
                    except Exception as e:
                        logger.error("[MSG] ❌ Send failed: %s", e)

    except Exception as e:
        logger.exception("[WEBHOOK] Error: %s", e)

    return {"status": "received"}


# ─────────────────────────────────────────────────────────────────────────────
# TEST SEND — verify token works
# Visit: /test-send?to=919969784982
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/test-send")
async def test_send(to: str):
    try:
        result = send_whatsapp_text(
            phone_number=to,
            message="✅ Test message — token and phone ID are working!",
        )
        return {"status": "sent ✅", "response": result}
    except Exception as e:
        return {"status": "failed ❌", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    conn     = sqlite3.connect(DB_FILE)
    pending  = conn.execute("SELECT COUNT(*) FROM reminders WHERE sent=0").fetchone()[0]
    total    = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
    upcoming = conn.execute(
        "SELECT phone, task, remind_at FROM reminders WHERE sent=0 ORDER BY remind_at LIMIT 5"
    ).fetchall()
    conn.close()

    return {
        "status"             : "ok ✅",
        "current_time_ist"   : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "groq_api_key"       : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "whatsapp_token"     : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "token_prefix"       : WHATSAPP_TOKEN[:12] + "..." if WHATSAPP_TOKEN else "MISSING",
        "phone_number_id"    : PHONE_NUMBER_ID or "MISSING ❌",
        "groq_sdk"           : "OK ✅" if groq_client else "ERROR ❌",
        "scheduler"          : "RUNNING ✅",
        "keepalive"          : "RUNNING ✅" if RENDER_URL else "NO URL SET ⚠️",
        "reminders_pending"  : pending,
        "reminders_total"    : total,
        "upcoming_reminders" : [{"task": r[1], "remind_at": r[2]} for r in upcoming],
        "users_in_memory"    : len(conversation_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    