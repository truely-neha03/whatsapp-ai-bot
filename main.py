"""
WhatsApp AI Bot - FastAPI + Groq (Free AI) + Meta WhatsApp Cloud API

Setup:
1. pip install fastapi uvicorn python-dotenv requests groq
2. Create .env file with your values
3. Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000

.env file:
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxx
WHATSAPP_TOKEN=EAAxxxxxxxxxxxxxxxxxxxxx
PHONE_NUMBER_ID=1200795349780072
VERIFY_TOKEN=my_secret_token_123
"""

import os
import json
import logging
from typing import Any, Dict

# ── Load .env ────────────────────────────────────────────────────────────────
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

GROQ_MODEL        = "llama-3.3-70b-versatile"   # Free, fast, works great
GRAPH_API_VERSION = "v25.0"

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("whatsapp-bot")

# ── Startup Diagnostics ──────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("STARTUP DIAGNOSTICS")
logger.info("  VERIFY_TOKEN    : %s", VERIFY_TOKEN)
logger.info("  PHONE_NUMBER_ID : %s", PHONE_NUMBER_ID or "(MISSING ❌)")
logger.info("  WHATSAPP_TOKEN  : %s", "SET ✅" if WHATSAPP_TOKEN else "(MISSING ❌)")
logger.info("  GROQ_API_KEY    : %s", "SET ✅" if GROQ_API_KEY else "(MISSING ❌)")
logger.info("  GROQ_MODEL      : %s", GROQ_MODEL)
logger.info("=" * 60)

# ── Import Groq ───────────────────────────────────────────────────────────────
groq_client = None
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("Groq SDK : OK ✅")
except ImportError:
    logger.error("Groq SDK NOT installed!")
    logger.error("Run: pip install groq")

# ── Conversation Memory ───────────────────────────────────────────────────────
# Remembers last 10 messages per user phone number
conversation_history: Dict[str, list] = {}

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="WhatsApp AI Bot")


# ─────────────────────────────────────────────────────────────────────────────
# AI REPLY USING GROQ
# ─────────────────────────────────────────────────────────────────────────────
def generate_ai_reply(prompt: str, sender: str) -> str:
    """
    Send message to Groq LLaMA and return reply.
    Remembers conversation history per user.
    Groq is free, fast, and works in India without any billing.
    """
    logger.info("[AI] Message from %s: %s", sender, prompt[:120])

    # Guards
    if not GROQ_API_KEY:
        logger.error("[AI] GROQ_API_KEY missing in .env")
        return "Sorry, AI is not configured."

    if groq_client is None:
        logger.error("[AI] Groq SDK not installed. Run: pip install groq")
        return "Sorry, AI SDK is missing."

    try:
        # Get or create conversation history for this sender
        if sender not in conversation_history:
            conversation_history[sender] = []
            logger.info("[AI] New user — starting fresh conversation")

        # Add user message to history
        conversation_history[sender].append({
            "role": "user",
            "content": prompt
        })

        # Keep only last 10 messages to avoid token limits
        recent = conversation_history[sender][-10:]

        # Call Groq
        logger.info("[AI] Calling %s...", GROQ_MODEL)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful and friendly WhatsApp assistant.
- Keep replies short and clear (under 100 words)
- Be conversational and warm
- Reply in the same language the user writes in
- If asked to set a reminder, confirm it clearly
- Never make up false information
- Be concise — this is WhatsApp, not email"""
                },
                *recent
            ],
            max_tokens=300,
            temperature=0.7,
        )

        reply = response.choices[0].message.content.strip()
        logger.info("[AI] ✅ Reply: %s", reply[:200])

        # Save assistant reply to history
        conversation_history[sender].append({
            "role": "assistant",
            "content": reply
        })

        return reply

    except Exception as e:
        error_msg = str(e)
        logger.exception("[AI] Error: %s", error_msg)

        if "401" in error_msg or "invalid_api_key" in error_msg:
            logger.error("[AI] Invalid Groq API key! Check GROQ_API_KEY in .env")
            return "Sorry, AI configuration error."
        elif "429" in error_msg:
            logger.error("[AI] Groq rate limit hit — try again in a moment")
            return "Sorry, AI is busy. Please try again in a moment."
        elif "model_not_found" in error_msg:
            logger.error("[AI] Groq model not found!")
            return "Sorry, AI model error."
        else:
            return "Sorry, I encountered an error. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# SEND WHATSAPP MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
def send_whatsapp_text(
    phone_number: str,
    message: str,
    phone_number_id: str | None = None,
) -> dict:
    """
    Send a WhatsApp message via Meta Cloud API.
    Raises clear errors so we know exactly what went wrong.
    """
    effective_phone_id = phone_number_id or PHONE_NUMBER_ID

    # Guards
    if not WHATSAPP_TOKEN:
        raise ValueError("WHATSAPP_TOKEN missing in .env")
    if not effective_phone_id:
        raise ValueError("PHONE_NUMBER_ID missing in .env")

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{effective_phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message,
        },
    }

    logger.info("[SEND] Sending to %s...", phone_number)
    resp = requests.post(url, headers=headers, json=payload, timeout=10)
    logger.info("[SEND] Status: %s", resp.status_code)
    logger.info("[SEND] Response: %s", resp.text)

    if not resp.ok:
        raise RuntimeError(f"Meta API error {resp.status_code}: {resp.text}")

    logger.info("[SEND] ✅ Message sent successfully!")
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# WEBHOOK VERIFICATION (GET /webhook)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/webhook", response_class=PlainTextResponse)
async def verify_webhook(request: Request):
    """
    Meta sends hub.mode, hub.verify_token, hub.challenge.
    We return hub.challenge when token matches.
    """
    params       = dict(request.query_params)
    mode         = params.get("hub.mode") or params.get("mode")
    verify_token = params.get("hub.verify_token") or params.get("verify_token")
    challenge    = params.get("hub.challenge") or params.get("challenge")

    logger.info("[VERIFY] mode=%s token_match=%s", mode, verify_token == VERIFY_TOKEN)

    if verify_token == VERIFY_TOKEN and challenge:
        logger.info("[VERIFY] ✅ Webhook verified!")
        return PlainTextResponse(content=challenge, status_code=200)

    logger.warning("[VERIFY] ❌ Token mismatch")
    raise HTTPException(status_code=403, detail="Verification token mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# RECEIVE MESSAGES (POST /webhook)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/webhook")
async def receive_message(payload: Dict[str, Any]):
    """
    Receives incoming WhatsApp messages and sends AI reply back.
    Always returns 200 so Meta does not retry.
    """
    logger.info("[WEBHOOK] Payload:\n%s", json.dumps(payload, indent=2))

    try:
        entries = payload.get("entry", [])

        if not entries:
            logger.info("[WEBHOOK] No entries — test ping, ignoring")
            return {"status": "ignored"}

        for entry in entries:
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Skip delivery receipts and read receipts
                if "statuses" in value and "messages" not in value:
                    logger.info("[WEBHOOK] Status update — skipping")
                    continue

                messages = value.get("messages") or []
                if not messages:
                    logger.info("[WEBHOOK] No messages — skipping")
                    continue

                # Get phone number ID from webhook metadata
                metadata_phone_id = value.get("metadata", {}).get("phone_number_id")
                phone_id_to_use   = metadata_phone_id or PHONE_NUMBER_ID
                logger.info("[WEBHOOK] Phone ID: %s", phone_id_to_use)

                for message in messages:
                    try:
                        sender   = message.get("from")
                        msg_type = message.get("type")

                        logger.info("[MSG] from=%s type=%s", sender, msg_type)

                        if not sender:
                            logger.warning("[MSG] No sender — skipping")
                            continue

                        # Handle different message types
                        if msg_type == "text":
                            text_body = message.get("text", {}).get("body", "")
                            logger.info("[MSG] Text: %s", text_body)
                            reply = generate_ai_reply(text_body, sender)

                        elif msg_type == "image":
                            reply = "I received your image! I can only process text messages for now. 😊"

                        elif msg_type == "audio":
                            reply = "I received your voice message! I can only process text messages for now. 😊"

                        elif msg_type == "document":
                            reply = "I received your document! I can only process text messages for now. 😊"

                        else:
                            logger.info("[MSG] Unsupported type: %s — skipping", msg_type)
                            continue

                        # Send reply back on WhatsApp
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
                                logger.error("[MSG] ❌ Token expired! Regenerate at developers.facebook.com")
                        except Exception as e:
                            logger.exception("[MSG] Unexpected send error: %s", e)

                    except Exception as e:
                        logger.exception("[MSG] Failed processing message: %s", e)

    except Exception as e:
        logger.exception("[WEBHOOK] Failed processing payload: %s", e)

    # Always return 200
    return {"status": "received"}


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Visit http://localhost:8000/health to check everything is working.
    All values should show SET or OK — nothing should show MISSING or ERROR.
    """
    return {
        "status"          : "ok ✅",
        "groq_api_key"    : "SET ✅" if GROQ_API_KEY else "MISSING ❌",
        "groq_model"      : GROQ_MODEL,
        "whatsapp_token"  : "SET ✅" if WHATSAPP_TOKEN else "MISSING ❌",
        "phone_number_id" : PHONE_NUMBER_ID or "MISSING ❌",
        "verify_token"    : VERIFY_TOKEN,
        "groq_sdk"        : "OK ✅" if groq_client else "ERROR ❌",
        "users_in_memory" : len(conversation_history),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)