r"""
Smoke test Vertex AI - Gemini models

Run with:  uv run hackathon\gemini\check_gemini.py
"""

import os

from dotenv import load_dotenv
from google import genai

# load env variables
load_dotenv()

PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

if not PROJECT:
    raise ValueError("GOOGLE_CLOUD_PROJECT ist nicht in der .env-Datei gesetzt.")


MODEL = "gemini-3.1-pro-preview"

client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)
resp = client.models.generate_content(
    model=MODEL,
    contents="Reply with one short fact about AI hallucinations.",
)

print(resp.text)

usage = resp.usage_metadata
tokens = usage.total_token_count if usage else "unknown"
print(f"\n{MODEL} via Vertex AI ({PROJECT}) | {tokens} tokens")
