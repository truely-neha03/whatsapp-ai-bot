"""
WhatsApp AI Bot - Complete Feature Set
Features: Reminders, Follow-ups, Voice Messages, Broadcast, Dashboard
"""

import os, json, logging, sqlite3, threading, time, re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse, HTMLResponse
import requests

# ── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN", "")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID", "")
VERIFY_TOKEN      = os.getenv("VERIFY_TOKEN", "my_secret_token_123")
RENDER_URL        = os.getenv("RENDER_EXTERNAL_URL", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")  # for Whisper voice transcription
GROQ_MODEL        = "llama-3.3-70b-versatile"
GRAPH_API_VERSION = "v25.0"
IST               = ZoneInfo("Asia/Kolkata")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("whatsapp-bot")

logger.info("="*60)
logger.info("STARTUP DIAGNOSTICS")
logger.info("  WHATSAPP_TOKEN  : %s", "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌")
logger.info("  GROQ_API_KEY    : %s", "SET ✅" if GROQ_API_KEY else "MISSING ❌")
logger.info("  OPENAI_API_KEY  : %s", "SET ✅ (voice enabled)" if OPENAI_API_KEY else "NOT SET (voice disabled)")
logger.info("  CURRENT IST     : %s", datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"))
logger.info("="*60)

groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq SDK : OK ✅")
except ImportError:
    logger.error("Groq SDK not installed. Run: pip install groq")

conversation_history: Dict[str, list] = {}
app = FastAPI(title="WhatsApp AI Bot")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
DB_FILE = "bot.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)

    # Reminders — full schema with status + history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            reminder_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          TEXT NOT NULL,
            reminder_text    TEXT NOT NULL,
            reminder_type    TEXT DEFAULT 'one-time',
            schedule_time    TEXT NOT NULL,
            recurrence_rule  TEXT DEFAULT '',
            recur_time       TEXT DEFAULT '',
            recur_day        TEXT DEFAULT '',
            status           TEXT DEFAULT 'pending',
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at       TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Follow-ups
    conn.execute("""
        CREATE TABLE IF NOT EXISTS followups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            phone       TEXT NOT NULL,
            contact     TEXT NOT NULL,
            reason      TEXT DEFAULT '',
            followup_at TEXT NOT NULL,
            status      TEXT DEFAULT 'pending',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Broadcast messages
    conn.execute("""
        CREATE TABLE IF NOT EXISTS broadcasts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            sender      TEXT NOT NULL,
            message     TEXT NOT NULL,
            recipients  TEXT NOT NULL,
            sent_count  INTEGER DEFAULT 0,
            fail_count  INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'pending',
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Broadcast delivery tracking
    conn.execute("""
        CREATE TABLE IF NOT EXISTS broadcast_delivery (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            broadcast_id INTEGER NOT NULL,
            phone        TEXT NOT NULL,
            status       TEXT DEFAULT 'pending',
            sent_at      TEXT DEFAULT ''
        )
    """)

    conn.commit()
    conn.close()
    logger.info("Database initialised ✅")

# ── Reminder functions ────────────────────────────────────────────────────────
def save_reminder(user_id, text, schedule_time, reminder_type="one-time",
                  recurrence_rule="", recur_time="", recur_day=""):
    if schedule_time.tzinfo is None:
        schedule_time = schedule_time.replace(tzinfo=IST)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO reminders
        (user_id,reminder_text,reminder_type,schedule_time,recurrence_rule,recur_time,recur_day,status)
        VALUES (?,?,?,?,?,?,?,'pending')
    """, (user_id, text, reminder_type,
          schedule_time.strftime("%Y-%m-%d %H:%M:%S"),
          recurrence_rule, recur_time, recur_day))
    conn.commit()
    conn.close()

def get_due_reminders():
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""
        SELECT reminder_id,user_id,reminder_text,schedule_time,
               reminder_type,recurrence_rule,recur_time,recur_day
        FROM reminders WHERE schedule_time<=? AND status='pending'
        ORDER BY schedule_time
    """, (now,)).fetchall()
    conn.close()
    return rows

def update_reminder_status(rid, status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE reminders SET status=?,updated_at=CURRENT_TIMESTAMP WHERE reminder_id=?",
                 (status, rid))
    conn.commit()
    conn.close()

def reschedule_reminder(rid, recurrence_rule, recur_time, recur_day):
    now = datetime.now(IST)
    if recurrence_rule == "hourly":
        next_dt = now + timedelta(hours=1)
    elif recurrence_rule == "daily":
        try:
            h, m = map(int, recur_time.split(":"))
            next_dt = (now + timedelta(days=1)).replace(hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            next_dt = now + timedelta(days=1)
    elif recurrence_rule == "weekly":
        days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        try:
            target = days.index(recur_day.lower())
            h, m = map(int, recur_time.split(":"))
            ahead = target - now.weekday()
            if ahead <= 0:
                ahead += 7
            next_dt = (now + timedelta(days=ahead)).replace(hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            next_dt = now + timedelta(weeks=1)
    elif recurrence_rule == "monthly":
        try:
            h, m = map(int, recur_time.split(":"))
            month = now.month + 1 if now.month < 12 else 1
            year  = now.year if now.month < 12 else now.year + 1
            next_dt = now.replace(year=year, month=month, hour=h, minute=m, second=0, microsecond=0)
        except Exception:
            next_dt = now + timedelta(days=30)
    else:
        return
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""UPDATE reminders SET schedule_time=?,status='pending',updated_at=CURRENT_TIMESTAMP
                    WHERE reminder_id=?""",
                 (next_dt.strftime("%Y-%m-%d %H:%M:%S"), rid))
    conn.commit()
    conn.close()

def get_user_reminders(user_id, status_filter=None):
    conn = sqlite3.connect(DB_FILE)
    if status_filter:
        rows = conn.execute("""
            SELECT reminder_id,reminder_text,schedule_time,reminder_type,recurrence_rule,status
            FROM reminders WHERE user_id=? AND status=? ORDER BY schedule_time
        """, (user_id, status_filter)).fetchall()
    else:
        rows = conn.execute("""
            SELECT reminder_id,reminder_text,schedule_time,reminder_type,recurrence_rule,status
            FROM reminders WHERE user_id=? ORDER BY schedule_time DESC LIMIT 20
        """, (user_id,)).fetchall()
    conn.close()
    return rows

def cancel_reminder(rid, user_id):
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.execute("""UPDATE reminders SET status='cancelled',updated_at=CURRENT_TIMESTAMP
                           WHERE reminder_id=? AND user_id=?""", (rid, user_id))
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed

# ── Follow-up functions ───────────────────────────────────────────────────────
def save_followup(phone, contact, reason, followup_at):
    if followup_at.tzinfo is None:
        followup_at = followup_at.replace(tzinfo=IST)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO followups (phone,contact,reason,followup_at,status) VALUES (?,?,?,?,'pending')",
                 (phone, contact, reason, followup_at.strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def get_due_followups():
    now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""SELECT id,phone,contact,reason FROM followups
                           WHERE followup_at<=? AND status='pending'""", (now,)).fetchall()
    conn.close()
    return rows

def update_followup_status(fid, status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE followups SET status=? WHERE id=?", (status, fid))
    conn.commit()
    conn.close()

def get_user_followups(phone):
    conn = sqlite3.connect(DB_FILE)
    rows = conn.execute("""SELECT id,contact,reason,followup_at,status FROM followups
                           WHERE phone=? AND status='pending' ORDER BY followup_at LIMIT 10""",
                        (phone,)).fetchall()
    conn.close()
    return rows

def cancel_followup(fid, phone):
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.execute("UPDATE followups SET status='cancelled' WHERE id=? AND phone=?", (fid, phone))
    conn.commit()
    changed = cur.rowcount > 0
    conn.close()
    return changed

# ── Broadcast functions ───────────────────────────────────────────────────────
def save_broadcast(sender, message, recipients: List[str]):
    conn = sqlite3.connect(DB_FILE)
    cur  = conn.execute("""INSERT INTO broadcasts (sender,message,recipients,status)
                           VALUES (?,?,?,'sending')""",
                        (sender, message, json.dumps(recipients)))
    broadcast_id = cur.lastrowid
    for phone in recipients:
        conn.execute("INSERT INTO broadcast_delivery (broadcast_id,phone,status) VALUES (?,?,'pending')",
                     (broadcast_id, phone))
    conn.commit()
    conn.close()
    return broadcast_id

def update_broadcast_delivery(broadcast_id, phone, status):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""UPDATE broadcast_delivery SET status=?,sent_at=CURRENT_TIMESTAMP
                    WHERE broadcast_id=? AND phone=?""", (status, broadcast_id, phone))
    if status == "sent":
        conn.execute("UPDATE broadcasts SET sent_count=sent_count+1 WHERE id=?", (broadcast_id,))
    else:
        conn.execute("UPDATE broadcasts SET fail_count=fail_count+1 WHERE id=?", (broadcast_id,))
    conn.commit()
    conn.close()

def finalize_broadcast(broadcast_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE broadcasts SET status='completed' WHERE id=?", (broadcast_id,))
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
        raise ValueError("WHATSAPP_TOKEN empty")
    if not pid:
        raise ValueError("PHONE_NUMBER_ID empty")
    url     = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{pid}/messages"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"messaging_product":"whatsapp","to":phone_number,"type":"text",
               "text":{"preview_url":False,"body":message}}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error: {e}") from e
    logger.info("[SEND] %s → HTTP %s", phone_number, resp.status_code)
    if resp.status_code == 401:
        raise RuntimeError("401 Token expired! Regenerate at developers.facebook.com")
    if not resp.ok:
        raise RuntimeError(f"Meta API {resp.status_code}: {resp.text}")
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# VOICE TRANSCRIPTION (Whisper via OpenAI)
# ─────────────────────────────────────────────────────────────────────────────
def transcribe_voice(audio_url: str) -> str:
    """Download WhatsApp audio and transcribe using Whisper"""
    if not OPENAI_API_KEY:
        return ""
    try:
        # Download audio from WhatsApp
        headers  = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
        audio_resp = requests.get(audio_url, headers=headers, timeout=30)
        if not audio_resp.ok:
            logger.error("[WHISPER] Failed to download audio: %s", audio_resp.status_code)
            return ""

        # Send to Whisper
        files = {"file": ("audio.ogg", audio_resp.content, "audio/ogg")}
        data  = {"model": "whisper-1"}
        whisper_resp = requests.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files=files, data=data, timeout=30
        )
        if whisper_resp.ok:
            text = whisper_resp.json().get("text", "")
            logger.info("[WHISPER] Transcribed: %s", text[:100])
            return text
        else:
            logger.error("[WHISPER] API error: %s", whisper_resp.text)
            return ""
    except Exception as e:
        logger.error("[WHISPER] Error: %s", e)
        return ""

def get_whatsapp_media_url(media_id: str) -> str:
    """Get download URL for WhatsApp media"""
    try:
        resp = requests.get(
            f"https://graph.facebook.com/{GRAPH_API_VERSION}/{media_id}",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            timeout=10
        )
        if resp.ok:
            return resp.json().get("url", "")
    except Exception as e:
        logger.error("[MEDIA] Error getting URL: %s", e)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER — reminders + follow-ups every 30 seconds
# ─────────────────────────────────────────────────────────────────────────────
def scheduler():
    logger.info("[SCHEDULER] Started ✅")
    while True:
        # Fire reminders
        try:
            for row in get_due_reminders():
                rid, phone, text, schedule_time, rtype, recurrence_rule, recur_time, recur_day = row
                try:
                    badge = {"hourly":"🔁 Hourly","daily":"🔁 Daily","weekly":"🔁 Weekly","monthly":"🔁 Monthly"}.get(recurrence_rule,"")
                    msg   = f"⏰ *Reminder{' ('+badge+')' if badge else ''}:*\n{text}"
                    send_whatsapp_text(phone_number=phone, message=msg)
                    logger.info("[SCHEDULER] ✅ Reminder #%d: %s", rid, text[:50])
                    if recurrence_rule and recurrence_rule != "none":
                        reschedule_reminder(rid, recurrence_rule, recur_time, recur_day)
                    else:
                        update_reminder_status(rid, "completed")
                except Exception as e:
                    logger.error("[SCHEDULER] ❌ Reminder #%d: %s", rid, e)
        except Exception as e:
            logger.error("[SCHEDULER] Reminder loop error: %s", e)

        # Fire follow-ups
        try:
            for fid, phone, contact, reason in get_due_followups():
                try:
                    msg = (f"📋 *Follow-up Reminder!*\n\n"
                           f"👤 Contact: {contact}\n"
                           f"💬 {reason or 'Time to follow up!'}\n\n"
                           f"Reach out now! 💪")
                    send_whatsapp_text(phone_number=phone, message=msg)
                    update_followup_status(fid, "completed")
                    logger.info("[SCHEDULER] ✅ Follow-up #%d: %s", fid, contact)
                except Exception as e:
                    logger.error("[SCHEDULER] ❌ Follow-up #%d: %s", fid, e)
        except Exception as e:
            logger.error("[SCHEDULER] Follow-up loop error: %s", e)

        time.sleep(30)

threading.Thread(target=scheduler, daemon=True).start()
logger.info("[SCHEDULER] Background thread started ✅")


# ─────────────────────────────────────────────────────────────────────────────
# KEEP ALIVE
# ─────────────────────────────────────────────────────────────────────────────
def keep_alive():
    time.sleep(60)
    while True:
        if RENDER_URL:
            try:
                requests.get(f"{RENDER_URL}/health", timeout=10)
            except Exception:
                pass
        time.sleep(840)

threading.Thread(target=keep_alive, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# AI PARSERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_reminder_with_ai(message: str) -> dict | None:
    try:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""IST now: {now}
User: "{message}"
Reply ONLY valid JSON. One of:
One-time: {{"is_reminder":true,"task":"desc","remind_at":"YYYY-MM-DD HH:MM","recurring":"none","recur_time":"","recur_day":""}}
Hourly:   {{"is_reminder":true,"task":"desc","remind_at":"YYYY-MM-DD HH:MM","recurring":"hourly","recur_time":"","recur_day":""}}
Daily:    {{"is_reminder":true,"task":"desc","remind_at":"YYYY-MM-DD HH:MM","recurring":"daily","recur_time":"HH:MM","recur_day":""}}
Weekly:   {{"is_reminder":true,"task":"desc","remind_at":"YYYY-MM-DD HH:MM","recurring":"weekly","recur_time":"HH:MM","recur_day":"Monday"}}
Monthly:  {{"is_reminder":true,"task":"desc","remind_at":"YYYY-MM-DD HH:MM","recurring":"monthly","recur_time":"HH:MM","recur_day":""}}
Not:      {{"is_reminder":false}}"""
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=150, temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("[REMINDER_PARSER] %s", e)
        return None

def parse_followup_with_ai(message: str) -> dict | None:
    try:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""IST now: {now}
User: "{message}"
Reply ONLY valid JSON:
YES: {{"is_followup":true,"contact":"name","reason":"why","followup_at":"YYYY-MM-DD HH:MM"}}
NO:  {{"is_followup":false}}
Calculate followup_at from "after 3 days", "after 1 week" etc. Default time 10:00 if not specified."""
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=120, temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("[FOLLOWUP_PARSER] %s", e)
        return None

def parse_broadcast_with_ai(message: str) -> dict | None:
    """Extract broadcast message and recipient list from user message"""
    try:
        now = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        prompt = f"""User wants to send a broadcast/bulk message: "{message}"
Reply ONLY valid JSON:
{{"is_broadcast":true,"broadcast_message":"the message to send","recipients":["919999999999","918888888888"]}}
Or if not a broadcast: {{"is_broadcast":false}}
Extract phone numbers with country code from the message."""
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.1)
        raw = resp.choices[0].message.content.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        logger.error("[BROADCAST_PARSER] %s", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# BROADCAST SENDER
# ─────────────────────────────────────────────────────────────────────────────
def send_broadcast(sender: str, message: str, recipients: List[str]) -> str:
    """Send message to multiple recipients and track delivery"""
    broadcast_id = save_broadcast(sender, message, recipients)
    sent, failed = 0, 0

    for phone in recipients:
        try:
            send_whatsapp_text(phone_number=phone, message=f"📢 *Broadcast Message:*\n\n{message}")
            update_broadcast_delivery(broadcast_id, phone, "sent")
            sent += 1
            time.sleep(0.5)  # small delay to avoid rate limiting
        except Exception as e:
            logger.error("[BROADCAST] Failed to send to %s: %s", phone, e)
            update_broadcast_delivery(broadcast_id, phone, "failed")
            failed += 1

    finalize_broadcast(broadcast_id)
    return (f"📢 *Broadcast Complete!*\n\n"
            f"✅ Sent: {sent}\n"
            f"❌ Failed: {failed}\n"
            f"📊 Total: {len(recipients)}\n\n"
            f"Broadcast #{broadcast_id} saved in history.")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE AI REPLY
# ─────────────────────────────────────────────────────────────────────────────
REMINDER_KW  = ["remind","reminder","alert","notify","don't forget","every day","every week",
                "every month","every hour","daily","weekly","monthly","hourly","recurring","याद"]
FOLLOWUP_KW  = ["follow up","follow-up","followup","check with","check back","get back to","reach out","फॉलो अप"]
BROADCAST_KW = ["broadcast","send to all","send to everyone","bulk message","send to multiple","mass message"]
LIST_R_KW    = ["my reminders","list reminders","show reminders","all reminders"]
LIST_F_KW    = ["my followups","list followups","show followups","my follow"]
LIST_H_KW    = ["reminder history","completed reminders","past reminders"]
CANCEL_R_KW  = ["cancel reminder","delete reminder","remove reminder","stop reminder"]
CANCEL_F_KW  = ["cancel followup","delete followup","cancel follow"]

def generate_ai_reply(prompt: str, sender: str) -> str:
    logger.info("[AI] from=%s msg=%s", sender, prompt[:120])
    if not GROQ_API_KEY or groq_client is None:
        return "Sorry, AI is not configured."

    try:
        p = prompt.lower()

        # ── List pending reminders ────────────────────────────────────────────
        if any(kw in p for kw in LIST_R_KW):
            rows = get_user_reminders(sender, "pending")
            if not rows:
                return "You have no pending reminders. 😊\n\nSet one: _'Remind me tomorrow at 9am to call John'_"
            lines = ["📋 *Your Pending Reminders:*\n"]
            for rid, text, schedule_time, rtype, recurrence_rule, status in rows:
                badge = f" 🔁 {recurrence_rule.capitalize()}" if recurrence_rule and recurrence_rule not in ("","none") else ""
                lines.append(f"*#{rid}* — {text}\n⏰ {schedule_time}{badge}\n")
            lines.append("\nTo cancel: _'cancel reminder #ID'_")
            lines.append("To see history: _'reminder history'_")
            return "\n".join(lines)

        # ── Reminder history ──────────────────────────────────────────────────
        if any(kw in p for kw in LIST_H_KW):
            rows = get_user_reminders(sender, "completed")
            if not rows:
                return "No completed reminders yet."
            lines = ["✅ *Completed Reminders:*\n"]
            for rid, text, schedule_time, rtype, recurrence_rule, status in rows[:10]:
                lines.append(f"#{rid} — {text} (at {schedule_time})\n")
            return "\n".join(lines)

        # ── List follow-ups ───────────────────────────────────────────────────
        if any(kw in p for kw in LIST_F_KW):
            rows = get_user_followups(sender)
            if not rows:
                return "You have no pending follow-ups. 😊\n\nSet one: _'Follow up with Raj after 3 days'_"
            lines = ["📋 *Your Pending Follow-ups:*\n"]
            for fid, contact, reason, followup_at, status in rows:
                lines.append(f"*#{fid}* — {contact}\n💬 {reason or 'General follow-up'}\n⏰ {followup_at}\n")
            lines.append("To cancel: _'cancel followup #ID'_")
            return "\n".join(lines)

        # ── Cancel reminder ───────────────────────────────────────────────────
        if any(kw in p for kw in CANCEL_R_KW):
            match = re.search(r"#?(\d+)", prompt)
            if match:
                rid = int(match.group(1))
                if cancel_reminder(rid, sender):
                    return f"✅ Reminder #{rid} cancelled successfully!"
                return f"❌ Reminder #{rid} not found or already done."
            return "Please specify ID. Example: _'cancel reminder #3'_"

        # ── Cancel follow-up ──────────────────────────────────────────────────
        if any(kw in p for kw in CANCEL_F_KW):
            match = re.search(r"#?(\d+)", prompt)
            if match:
                fid = int(match.group(1))
                if cancel_followup(fid, sender):
                    return f"✅ Follow-up #{fid} cancelled!"
                return f"❌ Follow-up #{fid} not found."
            return "Please specify ID. Example: _'cancel followup #2'_"

        # ── Broadcast ─────────────────────────────────────────────────────────
        if any(kw in p for kw in BROADCAST_KW):
            phones = re.findall(r"9\d{9}", prompt)  # extract Indian numbers
            if phones:
                msg_match = re.search(r"['\"](.+)['\"]", prompt)
                if msg_match:
                    bcast_msg = msg_match.group(1)
                    return send_broadcast(sender, bcast_msg, phones)
            return (
                "To broadcast, say:\n"
                "_'broadcast \"Your message here\" to 919999999999, 918888888888'_\n\n"
                "Or visit /dashboard to send bulk messages."
            )

        # ── Set follow-up ─────────────────────────────────────────────────────
        if any(kw in p for kw in FOLLOWUP_KW):
            data = parse_followup_with_ai(prompt)
            if data and data.get("is_followup"):
                contact  = data.get("contact","")
                reason   = data.get("reason","")
                time_str = data.get("followup_at","")
                try:
                    aware_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
                    if aware_dt <= datetime.now(IST):
                        return "⚠️ That time is in the past! Please give a future time."
                    save_followup(phone=sender, contact=contact, reason=reason, followup_at=aware_dt)
                    formatted = aware_dt.strftime("%d %b %Y at %I:%M %p IST")
                    return (f"✅ *Follow-up set!*\n\n"
                            f"👤 Contact: {contact}\n"
                            f"💬 {reason or 'General follow-up'}\n"
                            f"⏰ Reminder: {formatted}\n\n"
                            f"I'll remind you then! 💪")
                except ValueError:
                    return "Sorry, couldn't understand that time.\nTry: _'Follow up with Raj after 3 days'_"

        # ── Set reminder ──────────────────────────────────────────────────────
        if any(kw in p for kw in REMINDER_KW):
            data = parse_reminder_with_ai(prompt)
            if data and data.get("is_reminder"):
                task       = data.get("task", prompt)
                time_str   = data.get("remind_at","")
                recurring  = data.get("recurring","none")
                recur_time = data.get("recur_time","")
                recur_day  = data.get("recur_day","")
                try:
                    aware_dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M").replace(tzinfo=IST)
                    if aware_dt <= datetime.now(IST):
                        return "⚠️ That time is in the past! Please give a future time."
                    rtype = "recurring" if recurring not in ("","none") else "one-time"
                    save_reminder(user_id=sender, text=task, schedule_time=aware_dt,
                                  reminder_type=rtype, recurrence_rule=recurring,
                                  recur_time=recur_time, recur_day=recur_day)
                    formatted = aware_dt.strftime("%d %b %Y at %I:%M %p IST")
                    if rtype == "one-time":
                        return (f"✅ *Reminder set!*\n"
                                f"📋 Task: {task}\n"
                                f"⏰ Time: {formatted}\n\n"
                                f"I'll message you then!")
                    else:
                        labels = {"hourly":"Every hour","daily":f"Every day at {recur_time}",
                                  "weekly":f"Every {recur_day} at {recur_time}","monthly":f"Monthly at {recur_time}"}
                        return (f"✅ *Recurring Reminder set!* 🔁\n"
                                f"📋 Task: {task}\n"
                                f"⏰ First: {formatted}\n"
                                f"🔄 Repeats: {labels.get(recurring, recurring)}\n\n"
                                f"To stop: _'cancel reminder #ID'_\n"
                                f"To see all: _'my reminders'_")
                except ValueError:
                    return "Couldn't understand the time. Try: _'Remind me tomorrow at 5pm to call John'_"

        # ── Help command ──────────────────────────────────────────────────────
        if p in ["help","hi","hello","hey","start","menu"]:
            return (
                "👋 *Hello! I'm your WhatsApp AI Assistant.*\n\n"
                "*🔔 Reminders:*\n"
                "• _'Remind me tomorrow at 9am to call John'_\n"
                "• _'Remind me every day at 8am to exercise'_\n"
                "• _'my reminders'_ — see all reminders\n"
                "• _'cancel reminder #3'_ — cancel one\n\n"
                "*📋 Follow-ups:*\n"
                "• _'Follow up with Raj after 3 days'_\n"
                "• _'my followups'_ — see all follow-ups\n\n"
                "*💬 Chat:*\n"
                "• Ask me anything!\n\n"
                "*🎤 Voice:*\n"
                "• Send voice messages — I'll understand them!"
            )

        # ── Normal AI reply ───────────────────────────────────────────────────
        history = conversation_history.setdefault(sender, [])
        history.append({"role":"user","content":prompt})
        recent = history[-10:]
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role":"system","content":(
                    "You are a helpful WhatsApp assistant.\n"
                    "- Keep replies under 100 words\n"
                    "- Be warm and conversational\n"
                    "- Reply in same language as user\n"
                    "- You can set reminders and follow-ups\n"
                    "- Never make up information"
                )},
                *recent,
            ],
            max_tokens=300, temperature=0.7)
        reply = resp.choices[0].message.content.strip()
        history.append({"role":"assistant","content":reply})
        return reply

    except Exception as e:
        logger.exception("[AI] Error: %s", e)
        return "Sorry, I encountered an error. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    p = dict(request.query_params)
    if p.get("hub.verify_token") == VERIFY_TOKEN and p.get("hub.challenge"):
        return PlainTextResponse(content=p["hub.challenge"], status_code=200)
    raise HTTPException(status_code=403, detail="Token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# RECEIVE MESSAGES (text + voice)
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

                    elif msg_type == "audio":
                        # Voice message — transcribe with Whisper if available
                        if OPENAI_API_KEY:
                            media_id  = msg.get("audio", {}).get("id", "")
                            media_url = get_whatsapp_media_url(media_id)
                            if media_url:
                                transcribed = transcribe_voice(media_url)
                                if transcribed:
                                    reply = f"🎤 _Transcribed: \"{transcribed}\"_\n\n"
                                    reply += generate_ai_reply(transcribed, sender)
                                else:
                                    reply = "Sorry, I couldn't transcribe your voice message. Please try typing."
                            else:
                                reply = "Sorry, couldn't access your voice message. Please try typing."
                        else:
                            reply = "🎤 Voice messages received! To enable transcription, add OPENAI_API_KEY to settings."

                    elif msg_type == "image":
                        reply = "I received your image! I can only process text and voice for now. 😊"
                    elif msg_type == "document":
                        reply = "I received your document! I can only process text and voice for now. 😊"
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
# BROADCAST API ENDPOINT
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/broadcast")
async def broadcast_api(request: Request):
    """
    API to send broadcast messages.
    Body: {"sender": "919999999999", "message": "Hello everyone!", "recipients": ["919999999999"]}
    """
    try:
        body       = await request.json()
        sender     = body.get("sender", "admin")
        message    = body.get("message", "")
        recipients = body.get("recipients", [])
        if not message or not recipients:
            raise HTTPException(status_code=400, detail="message and recipients required")
        result = send_broadcast(sender, message, recipients)
        return {"status": "ok", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD — Web UI to manage reminders
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    conn = sqlite3.connect(DB_FILE)
    pending_r   = conn.execute("SELECT reminder_id,user_id,reminder_text,schedule_time,reminder_type,recurrence_rule,status FROM reminders WHERE status='pending' ORDER BY schedule_time LIMIT 50").fetchall()
    completed_r = conn.execute("SELECT reminder_id,user_id,reminder_text,schedule_time,reminder_type,status FROM reminders WHERE status='completed' ORDER BY updated_at DESC LIMIT 20").fetchall()
    pending_f   = conn.execute("SELECT id,phone,contact,reason,followup_at,status FROM followups WHERE status='pending' ORDER BY followup_at LIMIT 20").fetchall()
    broadcasts  = conn.execute("SELECT id,sender,message,sent_count,fail_count,status,created_at FROM broadcasts ORDER BY created_at DESC LIMIT 10").fetchall()
    conn.close()

    def rows_to_html(rows, cols):
        if not rows:
            return "<tr><td colspan='100' style='text-align:center;color:#888'>No records</td></tr>"
        html = ""
        for row in rows:
            html += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        return html

    r_header = "<tr><th>ID</th><th>User</th><th>Task</th><th>Time</th><th>Type</th><th>Recurrence</th><th>Status</th></tr>"
    f_header = "<tr><th>ID</th><th>Phone</th><th>Contact</th><th>Reason</th><th>Time</th><th>Status</th></tr>"
    b_header = "<tr><th>ID</th><th>Sender</th><th>Message</th><th>Sent</th><th>Failed</th><th>Status</th><th>Created</th></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>WhatsApp Bot Dashboard</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  body{{font-family:Arial,sans-serif;margin:0;background:#f0f2f5}}
  .header{{background:#25D366;color:white;padding:20px;text-align:center}}
  .header h1{{margin:0;font-size:24px}}
  .stats{{display:flex;gap:15px;padding:20px;flex-wrap:wrap}}
  .stat{{background:white;border-radius:10px;padding:20px;flex:1;min-width:150px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1)}}
  .stat h2{{margin:0;font-size:32px;color:#25D366}}
  .stat p{{margin:5px 0 0;color:#666;font-size:14px}}
  .section{{background:white;margin:0 20px 20px;border-radius:10px;overflow:hidden;box-shadow:0 2px 5px rgba(0,0,0,0.1)}}
  .section h3{{margin:0;padding:15px 20px;background:#075E54;color:white;font-size:16px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{background:#f8f9fa;padding:10px;text-align:left;border-bottom:2px solid #dee2e6;color:#495057}}
  td{{padding:10px;border-bottom:1px solid #dee2e6;color:#333}}
  tr:hover td{{background:#f8f9fa}}
  .badge{{padding:3px 8px;border-radius:4px;font-size:11px;font-weight:bold}}
  .pending{{background:#fff3cd;color:#856404}}
  .completed{{background:#d4edda;color:#155724}}
  .cancelled{{background:#f8d7da;color:#721c24}}
  .refresh{{position:fixed;bottom:20px;right:20px;background:#25D366;color:white;border:none;padding:12px 20px;border-radius:25px;cursor:pointer;font-size:14px;box-shadow:0 3px 10px rgba(0,0,0,0.2)}}
  .ist{{font-size:12px;color:#aaa;margin-top:5px}}
</style>
</head>
<body>
<div class="header">
  <h1>📱 WhatsApp AI Bot Dashboard</h1>
  <div class="ist">Current IST: {datetime.now(IST).strftime('%d %b %Y %I:%M:%S %p')}</div>
</div>
<div class="stats">
  <div class="stat"><h2>{len(pending_r)}</h2><p>Pending Reminders</p></div>
  <div class="stat"><h2>{len(completed_r)}</h2><p>Completed Reminders</p></div>
  <div class="stat"><h2>{len(pending_f)}</h2><p>Pending Follow-ups</p></div>
  <div class="stat"><h2>{len(broadcasts)}</h2><p>Broadcasts Sent</p></div>
</div>

<div class="section">
  <h3>🔔 Pending Reminders ({len(pending_r)})</h3>
  <table>{r_header}{rows_to_html(pending_r, 7)}</table>
</div>

<div class="section">
  <h3>✅ Completed Reminders ({len(completed_r)})</h3>
  <table>{r_header}{rows_to_html(completed_r, 6)}</table>
</div>

<div class="section">
  <h3>📋 Pending Follow-ups ({len(pending_f)})</h3>
  <table>{f_header}{rows_to_html(pending_f, 6)}</table>
</div>

<div class="section">
  <h3>📢 Broadcast History ({len(broadcasts)})</h3>
  <table>{b_header}{rows_to_html(broadcasts, 7)}</table>
</div>

<button class="refresh" onclick="location.reload()">🔄 Refresh</button>
</body>
</html>"""
    return HTMLResponse(content=html)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    conn = sqlite3.connect(DB_FILE)
    r_pending = conn.execute("SELECT COUNT(*) FROM reminders WHERE status='pending'").fetchone()[0]
    r_total   = conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0]
    f_pending = conn.execute("SELECT COUNT(*) FROM followups WHERE status='pending'").fetchone()[0]
    b_total   = conn.execute("SELECT COUNT(*) FROM broadcasts").fetchone()[0]
    conn.close()
    return {
        "status"           : "ok ✅",
        "current_time_ist" : datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "groq_api_key"     : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "whatsapp_token"   : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "voice_transcription": "ENABLED ✅" if OPENAI_API_KEY else "DISABLED (add OPENAI_API_KEY)",
        "scheduler"        : "RUNNING ✅",
        "reminders_pending": r_pending,
        "reminders_total"  : r_total,
        "followups_pending": f_pending,
        "broadcasts_total" : b_total,
        "dashboard_url"    : f"{RENDER_URL}/dashboard" if RENDER_URL else "http://localhost:8000/dashboard",
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)