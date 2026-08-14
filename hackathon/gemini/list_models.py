r"""
List Gemini models available to Vertex AI project.

Run with:  uv run hackathon\gemini\list_models.py
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


client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

models = sorted(
    m.name.removeprefix("publishers/google/models/")
    for m in client.models.list(config={"query_base": True})
    if m.name
)

print(f"{len(models)} base models in {PROJECT} ({LOCATION}):\n")
for name in models:
    print(" ", name)
