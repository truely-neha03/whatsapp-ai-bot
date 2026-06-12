"""
WhatsApp AI Bot - FastAPI + Groq + Reminders (One-time + Recurring) + Meta WhatsApp Cloud API

.env file:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
WHATSAPP_TOKEN=EAAxxxxxxxxxxxxxxxxxxxxx
PHONE_NUMBER_ID=1200795349780072
VERIFY_TOKEN=my_secret_token_123
RENDER_EXTERNAL_URL=https://whatsapp-ai-bot-faia.onrender.com

Recurring reminder examples users can send:
- "Remind me every day at 9am to drink water"
- "Remind me every Monday at 10am to send report"
- "Remind me every hour to check emails"
- "Remind me every week on Friday at 5pm to submit timesheet"
"""

import os
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
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
logger.info("  CURRENT IST     : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
logger.info("  RENDER_URL      : %s", RENDER_URL or "NOT SET ⚠️")
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

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="WhatsApp AI Bot with Recurring Reminders")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# Recurring types: none | hourly | daily | weekly | monthly
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE = "reminders.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            phone         TEXT NOT NULL,
            task          TEXT NOT NULL,
            remind_at     TEXT NOT NULL,
            recurring     TEXT DEFAULT 'none',
            recur_time    TEXT DEFAULT '',
            recur_day     TEXT DEFAULT '',
            sent          INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Add recurring columns if old db exists without them
    try:
        conn.execute("ALTER TABLE reminders ADD COLUMN recurring TEXT DEFAULT 'none'")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE reminders ADD COLUMN recur_time TEXT DEFAULT ''")
    except Exception:
        pass
    try:
        conn.execute("ALTER TABLE reminders ADD COLUMN recur_day TEXT DEFAULT ''")
    except Exception:
        pass
    conn.commit()
    conn.close()
    logger.info("Database initialised ✅")

def save_reminder(
    phone: str,
    task: str,
    remind_at: datetime,
    recurring: str = "none",
    recur_time: str = "",
    recur_day: str = "",
):
    if remind_at.tzinfo is None:
        remind_at = remind_at.replace(tzinfo=IST)
    else:
        remind_at = remind_at.astimezone(IST)

    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        """INSERT INTO reminders
           (phone, task, remind_at, recurring, recur_time, recur_day, sent)
           VALUES (?, ?, ?, ?, ?, ?, 0)""",
        (phone, task, remind_at.strftime("%Y-%m-%d %H:%M:%S"),
         recurring, recur_time, recur_day),
    )
    conn.commit()
    conn.close()
    logger.info("[DB] Saved reminder: '%s' at %s recurring=%s", task, remind_at, recurring)

def get_due_reminders():
    now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    conn    = sqlite3.connect(DB_FILE)
    rows    = conn.execute(
        "SELECT id, phone, task, remind_at, recurring, recur_time, recur_day "
        "FROM reminders WHERE remind_at <= ? AND sent = 0 ORDER BY remind_at",
        (now_str,),
    ).fetchall()
    conn.close()
    return rows

def reschedule_reminder(reminder_id: int, recurring: str, recur_time: str, recur_day: str):
    """Calculate next fire time and update remind_at, reset sent=0"""
    now = datetime.now(IST)

    if recurring == "hourly":
        next_dt = now + timedelta(hours=1)

    elif recurring == "daily":
        # recur_time = "HH:MM"
        try:
            h, m = map(int, recur_time.split(":"))
            next_dt = now.replace(hour=h, minute=m, second=0, microsecond=0) + timedelta(days=1)
        except Exception:
            next_dt = now + timedelta(days=1)

    elif recurring == "weekly":
        # recur_day = "Monday", recur_time = "HH:MM"
        days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        try:
            target_day = days.index(recur_day.lower())
            h, m = map(int, recur_time.split(":"))
            days_ahead = target_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_dt = (now + timedelta(days=days_ahead)).replace(
                hour=h, minute=m, second=0, microsecond=0
            )
        except Exception:
            next_dt = now + timedelta(weeks=1)

    elif recurring == "monthly":
        # Same day next month
        try:
            h, m = map(int, recur_time.split(":"))
            if now.month == 12:
                next_dt = now.replace(year=now.year+1, month=1, hour=h, minute=m, second=0, microsecond=0)
            else:
                next_dt = now.replace(month=now.month+1, hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            next_dt = now + timedelta(days=30)
    else:
        return  # not recurring — don't reschedule

    next_str = next_dt.strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "UPDATE reminders SET remind_at = ?, sent = 0 WHERE id = ?",
        (next_str, reminder_id)
    )
    conn.commit()
    conn.close()
    logger.info("[DB] Rescheduled #%d to %s", reminder_id, next_str)

def mark_sent(reminder_id: int):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE reminders SET sent = 1 WHERE id = ?", (reminder_id,))
    conn.commit()
    conn.close()

def get_active_reminders(phone: str):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute(
        "SELECT id, task, remind_at, recurring FROM reminders "
        "WHERE phone = ? AND sent = 0 ORDER BY remind_at LIMIT 10",
        (phone,)
    ).fetchall()
    conn.close()
    return rows

def cancel_reminder(reminder_id: int, phone: str) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.execute(
        "UPDATE reminders SET sent = 1 WHERE id = ? AND phone = ?",
        (reminder_id, phone)
    )
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed

init_db()


# ─────────────────────────────────────────────────────────────────────────────
# SEND WHATSAPP MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp_text(phone_number: str, message: str, phone_number_id: str = "") -> dict:
    pid   = phone_number_id or PHONE_NUMBER_ID
    token = WHATSAPP_TOKEN

    if not token:
        raise ValueError("WHATSAPP_TOKEN empty — update on Render")
    if not pid:
        raise ValueError("PHONE_NUMBER_ID empty — update on Render")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to":   phone_number,
        "type": "text",
        "text": {"preview_url": False, "body": message},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}") from e

    logger.info("[SEND] %s → HTTP %s", phone_number, resp.status_code)

    if resp.status_code == 401:
        raise RuntimeError("401 Token expired! Regenerate at developers.facebook.com")
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

            for reminder_id, phone, task, remind_at, recurring, recur_time, recur_day in due:
                try:
                    # Build message with recurring badge
                    if recurring != "none":
                        badge = {
                            "hourly":  "🔁 Hourly",
                            "daily":   "🔁 Daily",
                            "weekly":  "🔁 Weekly",
                            "monthly": "🔁 Monthly",
                        }.get(recurring, "🔁")
                        msg = f"⏰ *Reminder ({badge}):* {task}"
                    else:
                        msg = f"⏰ *Reminder:* {task}"

                    send_whatsapp_text(phone_number=phone, message=msg)
                    logger.info("[SCHEDULER] ✅ Sent #%d: %s", reminder_id, task)

                    if recurring != "none":
                        reschedule_reminder(reminder_id, recurring, recur_time, recur_day)
                    else:
                        mark_sent(reminder_id)

                except Exception as e:
                    logger.error("[SCHEDULER] ❌ Failed #%d: %s", reminder_id, e)

        except Exception as e:
            logger.error("[SCHEDULER] ❌ Loop error: %s", e)

        time.sleep(30)

threading.Thread(target=reminder_scheduler, daemon=True).start()
logger.info("[SCHEDULER] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# KEEP ALIVE
# ─────────────────────────────────────────────────────────────────────────────
def keep_alive():
    time.sleep(60)
    while True:
        if RENDER_URL:
            try:
                resp = requests.get(f"{RENDER_URL}/health", timeout=10)
                logger.info("[KEEPALIVE] ✅ %s", resp.status_code)
            except Exception as e:
                logger.warning("[KEEPALIVE] ⚠️ %s", e)
        time.sleep(840)

threading.Thread(target=keep_alive, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# PARSE REMINDER WITH AI
# ─────────────────────────────────────────────────────────────────────────────
def parse_reminder_with_ai(message: str) -> dict | None:
    try:
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        prompt  = f"""Current date and time (IST): {now_str}

User message: "{message}"

Is this a reminder request? Reply ONLY with valid JSON.

If ONE-TIME reminder:
{{"is_reminder": true, "task": "description", "remind_at": "YYYY-MM-DD HH:MM", "recurring": "none", "recur_time": "", "recur_day": ""}}

If HOURLY recurring:
{{"is_reminder": true, "task": "description", "remind_at": "YYYY-MM-DD HH:MM", "recurring": "hourly", "recur_time": "", "recur_day": ""}}

If DAILY recurring:
{{"is_reminder": true, "task": "description", "remind_at": "YYYY-MM-DD HH:MM", "recurring": "daily", "recur_time": "HH:MM", "recur_day": ""}}

If WEEKLY recurring:
{{"is_reminder": true, "task": "description", "remind_at": "YYYY-MM-DD HH:MM", "recurring": "weekly", "recur_time": "HH:MM", "recur_day": "Monday"}}

If MONTHLY recurring:
{{"is_reminder": true, "task": "description", "remind_at": "YYYY-MM-DD HH:MM", "recurring": "monthly", "recur_time": "HH:MM", "recur_day": ""}}

If NOT a reminder:
{{"is_reminder": false}}

Rules:
- remind_at is the FIRST occurrence datetime
- recur_time is in 24-hour HH:MM format
- recur_day is the day name for weekly (Monday, Tuesday etc)
- Calculate exact datetime from current IST time
- remind_at must be in the future
- No markdown, no extra text, only JSON"""

        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
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
    "every day", "every week", "every month", "every hour",
    "daily", "weekly", "monthly", "hourly", "recurring",
    "याद", "याद दिलाओ", "रिमाइंडर",
]

LIST_KEYWORDS   = ["list reminders", "show reminders", "my reminders", "what reminders"]
CANCEL_KEYWORDS = ["cancel reminder", "delete reminder", "remove reminder", "stop reminder"]

def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] from=%s msg=%s", sender, prompt[:120])

    if not GROQ_API_KEY or groq_client is None:
        return "Sorry, AI is not configured."

    try:
        prompt_lower = prompt.lower()

        # ── List reminders ────────────────────────────────────────────────────
        if any(kw in prompt_lower for kw in LIST_KEYWORDS):
            reminders = get_active_reminders(sender)
            if not reminders:
                return "You have no active reminders. 😊"
            lines = ["📋 *Your Active Reminders:*\n"]
            for r in reminders:
                rid, task, remind_at, recurring = r
                badge = f" 🔁 {recurring.capitalize()}" if recurring != "none" else ""
                lines.append(f"#{rid} — {task}\n⏰ {remind_at}{badge}\n")
            lines.append("\nTo cancel: _'cancel reminder #ID'_")
            return "\n".join(lines)

        # ── Cancel reminder ───────────────────────────────────────────────────
        if any(kw in prompt_lower for kw in CANCEL_KEYWORDS):
            import re
            match = re.search(r"#?(\d+)", prompt)
            if match:
                rid = int(match.group(1))
                if cancel_reminder(rid, sender):
                    return f"✅ Reminder #{rid} cancelled successfully!"
                else:
                    return f"❌ Reminder #{rid} not found or already sent."
            return "Please specify the reminder ID. Example: _'cancel reminder #5'_"

        # ── Set reminder ──────────────────────────────────────────────────────
        if any(kw in prompt_lower for kw in REMINDER_KEYWORDS):
            logger.info("[AI] Reminder request detected")
            data = parse_reminder_with_ai(prompt)

            if data and data.get("is_reminder"):
                task       = data.get("task", prompt)
                time_str   = data.get("remind_at", "")
                recurring  = data.get("recurring", "none")
                recur_time = data.get("recur_time", "")
                recur_day  = data.get("recur_day", "")

                try:
                    naive_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M")
                    aware_dt = naive_dt.replace(tzinfo=IST)

                    if aware_dt <= datetime.now(IST):
                        return (
                            "⚠️ That time is already in the past!\n"
                            "Please give me a future time.\n"
                            "Example: _'Remind me tomorrow at 9am to check emails'_"
                        )

                    save_reminder(
                        phone=sender,
                        task=task,
                        remind_at=aware_dt,
                        recurring=recurring,
                        recur_time=recur_time,
                        recur_day=recur_day,
                    )

                    formatted = aware_dt.strftime("%d %b %Y at %I:%M %p IST")

                    if recurring == "none":
                        return (
                            f"✅ *Reminder set!*\n"
                            f"📋 Task: {task}\n"
                            f"⏰ Time: {formatted}\n\n"
                            f"I'll message you then!"
                        )
                    else:
                        recur_labels = {
                            "hourly":  "Every hour",
                            "daily":   f"Every day at {recur_time}",
                            "weekly":  f"Every {recur_day} at {recur_time}",
                            "monthly": f"Every month at {recur_time}",
                        }
                        label = recur_labels.get(recurring, recurring)
                        return (
                            f"✅ *Recurring Reminder set!* 🔁\n"
                            f"📋 Task: {task}\n"
                            f"⏰ First: {formatted}\n"
                            f"🔄 Repeats: {label}\n\n"
                            f"To stop: _'cancel reminder #ID'_\n"
                            f"To see all: _'list reminders'_"
                        )

                except ValueError:
                    return (
                        "Sorry, I couldn't understand that time.\n"
                        "Try: _'Remind me every day at 9am to drink water'_"
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
                        "You are a helpful WhatsApp assistant.\n"
                        "- Keep replies under 100 words\n"
                        "- Be warm and conversational\n"
                        "- Reply in same language as user\n"
                        "- You can set one-time and recurring reminders\n"
                        "- Never make up information"
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

    raise HTTPException(status_code=403, detail="Token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# RECEIVE MESSAGES
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(payload: Dict[str, Any]):
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
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    conn     = sqlite3.connect(DB_FILE)
    pending  = conn.execute("SELECT COUNT(*) FROM reminders WHERE sent=0").fetchone()[0]
    total    = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
    upcoming = conn.execute(
        "SELECT phone, task, remind_at, recurring FROM reminders "
        "WHERE sent=0 ORDER BY remind_at LIMIT 5"
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
        "upcoming"          : [
            {"task": r[1], "remind_at": r[2], "recurring": r[3]} for r in upcoming
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)