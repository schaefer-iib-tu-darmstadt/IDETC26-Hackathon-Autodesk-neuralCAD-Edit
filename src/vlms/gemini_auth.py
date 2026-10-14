"""One place that decides how Gemini is reached.

Two code paths talk to Gemini -- the stock CadQuery-script harness (`src/vlms/gemini.py`, through
`google.genai.Client`) and the ADK agent (`src/agents/cad_agent_base.py`, through
`google.adk.models.google_llm.Gemini`). Both used to carry their own copy of the credential rule,
with different defaults. They now share this one.

Resolution order, first match wins:

1. `GOOGLE_CLOUD_PROJECT` set        -> Vertex AI with application-default credentials (ADC).
                                       `GOOGLE_CLOUD_LOCATION` is honoured, default "global".
                                       Needs `gcloud auth application-default login` or
                                       `GOOGLE_APPLICATION_CREDENTIALS` -- that is checked lazily
                                       by the SDK on the first request, not here.
2. `GOOGLE_API_KEY` or `GEMINI_API_KEY` -> the public Gemini API (the SDK's own precedence).
                                       `vertexai=False` is pinned so a stray
                                       `GOOGLE_GENAI_USE_VERTEXAI` in the environment cannot
                                       silently flip the backend; Vertex AI "Express mode"
                                       keys are therefore not supported -- use a project.
3. neither                           -> `ValueError` at construction time.

Variables are read from the process environment; a `.env` file in the repository root (and only
there) is loaded once on import so both callers see the same values.
"""

import os
from pathlib import Path

from google.genai import types

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:  # Credentials live in .env; load it once, here, for every caller.
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def resolve_gemini_client_kwargs(timeout_seconds):
    """Return the kwargs for `google.genai.Client(**kwargs)` for the configured backend.

    The same dict is what the ADK `Gemini(client_kwargs=...)` model forwards to its own client,
    so one resolver serves both the harness and the agent.
    """
    client_kwargs = {
        "http_options": types.HttpOptions(timeout=int(float(timeout_seconds) * 1000)),
    }
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    # Same precedence as the google-genai SDK itself, so its own log line and ours agree.
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if project:
        client_kwargs.update(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
    elif api_key:
        client_kwargs.update(vertexai=False, api_key=api_key)
    else:
        raise ValueError(
            "No Gemini credentials found. Set GOOGLE_CLOUD_PROJECT for Vertex AI (with "
            "application-default credentials), or GEMINI_API_KEY / GOOGLE_API_KEY for the "
            "public Gemini API. Both can live in a .env file in the repository root."
        )
    return client_kwargs


def describe_backend(client_kwargs):
    """Human-readable one-liner for logs and the credentials smoke test.

    Never contains a whole key: for the public API it shows at most the last four characters, so
    a log line can tell two keys apart without exposing either.
    """
    if client_kwargs.get("vertexai"):
        return (f"Vertex AI, project={client_kwargs.get('project')}, "
                f"location={client_kwargs.get('location')}")
    key = client_kwargs.get("api_key") or ""
    return f"Gemini API, api_key=...{key[-4:]}" if key else "Gemini API"
