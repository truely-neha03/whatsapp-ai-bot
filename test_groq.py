print("Script started...")
from dotenv import load_dotenv
load_dotenv()
import os
from groq import Groq

key = os.getenv("GROQ_API_KEY", "")
print(f"Key loaded: {key[:10]}...")

client = Groq(api_key=key)
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("✅ Groq working!")
print("Reply:", response.choices[0].message.content)