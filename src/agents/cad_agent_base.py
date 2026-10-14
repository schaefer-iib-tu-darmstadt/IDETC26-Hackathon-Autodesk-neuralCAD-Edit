"""Shared runner base for the ADK CAD edit agent.

Everything in here is independent of what the agent's tools are: the Gemini model object, the
generation config, the sync entry point the harness calls, the helpers that turn files into prompt
parts, the transport-retry classifier, and the fixed prompt fragments the loop sends between turns.
`cad_agent_v4.py` subclasses `CadAgentRunnerBase` and supplies `_run_edit`, i.e. the tools and the
loop around them.

The strings and defaults here are part of the agent's recorded behaviour -- `NUDGE`, `RESUME`,
`DONE_MARKERS`, the face cap in `_for_prompt` and the generation defaults all reach the model or
decide when the loop stops. Change them as a prompt change, not as a tidy-up. Known warts, all
kept on purpose for that reason (every one of these strings reaches the model verbatim):

* `NUDGE` tells the model to "call run_script", which was the edit tool of an earlier agent
  version; v4's edit tool is `apply_edit`. The nudge is only sent when nothing has been committed
  after a turn, which in the recorded runs happened rarely.
* `DONE_MARKERS` does not contain the bare word "DONE" that the v4 prompt asks for; v4 ends a run
  on its `finish` tool instead, so the markers are only a fallback.
* `_for_prompt`'s truncation note says "Call `inspect`", the lookup tool of the earlier agent;
  v4 has no `inspect`, its lookup tool is `ground`. The note is in the first message of every
  part with more than 60 faces.
* Two strings in `cad_git` have the same history: `face_delta` says "Use `inspect` for the rest"
  when an edit creates more than 20 faces, and the origin-identity refusal in `CadRepo._commit`
  tells the model to "call `revert`", which v4 does not expose either (the only revert is the
  automatic one after a failed contract).
"""

import asyncio
import mimetypes
import os
import os.path as osp

from google.adk.models.google_llm import Gemini
from google.genai import types

from src.vlms.gemini_auth import resolve_gemini_client_kwargs

SAFETY_OFF = [
    types.SafetySetting(category=category, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for category in (
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    )
]
DONE_MARKERS = ("EDIT_COMPLETE", "TASK_COMPLETE", "FINAL_ANSWER")


def _image_part(path):
    """Load an image as a Gemini inline-data part."""
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return types.Part(inline_data=types.Blob(data=handle.read(), mime_type=mime))


# Bugs in our own code, not transport trouble -- retrying them just burns the clock.
_PERMANENT_ERRORS = (TypeError, AttributeError, KeyError, IndexError, NameError,
                     ImportError, SyntaxError, ValueError)


def _is_transient(exc):
    """Return whether a transport-level failure is safe to retry."""
    if isinstance(exc, _PERMANENT_ERRORS):
        return False
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(k in blob for k in (
        "429", "500", "502", "503", "504", "resource_exhausted", "resource exhausted",
        "unavailable", "deadline", "timeout", "timed out", "connection", "reset",
        "internal error", "overloaded", "retry",
    ))


NUDGE = (
    "Nothing has been committed yet -- no valid, changed geometry exists. Write the edit as a "
    "CadQuery script and call run_script."
)
MAX_NUDGES = 2

RESUME = (
    "The previous request to you failed in transport, not in your work. Nothing you did was lost: "
    "the history and every checkpoint are intact. Continue from where you stopped."
)
# ADK can re-raise exhausted provider retries. Match status text because google-genai uses generic
# ServerError/ClientError classes rather than one exception type per HTTP status.
MAX_TRANSPORT_RETRIES = 3


# The repository summarises with max_faces=400 because preservation matching scores every face.
# The model does not need 400: on the dataset's largest part that is thousands of lines of JSON in
# the same message as the request and seven images. Measured on a multi-solid rotor request: the
# agent spent four of ten turns re-deriving by hand facts that sat in that wall of text.
# Trimming the face list for the prompt while the gate keeps the full one costs nothing.
PROMPT_MAX_FACES = 60


def _write_stdout(work_dir, text):
    """Persist a script's stdout next to its source. Best-effort: diagnostics never fail a run."""
    if not text:
        return
    try:
        os.makedirs(work_dir, exist_ok=True)
        with open(osp.join(work_dir, "stdout.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.write(text)
    except OSError:
        pass


def _for_prompt(summary):
    """Model-facing view of a repository summary: same structure, shorter face list."""
    if not summary:
        return summary
    trimmed = dict(summary)
    faces = trimmed.get("faces_listed") or []
    if len(faces) > PROMPT_MAX_FACES:
        trimmed["faces_listed"] = faces[:PROMPT_MAX_FACES]
        trimmed["faces_truncated"] = (
            f"Showing the {PROMPT_MAX_FACES} largest of {len(faces)} faces, sorted by area. Every "
            f"face carries the index of the solid it belongs to, and `solids[]` lists the full "
            f"face-type histogram per solid, so a face missing here is still accounted for. Call "
            f"`inspect` to see any of the remaining {len(faces) - PROMPT_MAX_FACES}."
        )
    return trimmed


class CadAgentRunnerBase:
    """Model, generation config and sync entry point. Subclasses implement `_run_edit`."""

    def __init__(self, config):
        self.config = config
        self.features = config.get("features") or {}

    def _feat(self, name, default):
        return self.features.get(name, default)

    def _model(self):
        """The ADK model object. Backend and credentials come from `src.vlms.gemini_auth`."""
        timeout = float(self.config.get("timeout_seconds", 300.0))
        client_kwargs = resolve_gemini_client_kwargs(timeout)
        return Gemini(model=self.config["model"], client_kwargs=client_kwargs)

    def _generation_config(self):
        kwargs = {
            "temperature": float(self.config.get("temperature", 0.0)),
            "max_output_tokens": int(self.config.get("max_output_tokens", 32768)),
            "safety_settings": SAFETY_OFF,
        }
        level = self.config.get("thinking_level")
        if level in ("low", "medium", "high"):
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=level,
                                                             include_thoughts=True)
        elif level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=-1,
                                                             include_thoughts=True)
        return types.GenerateContentConfig(**kwargs)

    def run_edit(self, instruction_text, task_info_dict, output_dir, max_turns=10,
                 harness_script_file=None):
        """Sync entry point for the harness, which runs tasks in threads."""
        return asyncio.run(self._run_edit(
            instruction_text, task_info_dict, output_dir, max_turns, harness_script_file
        ))

    async def _run_edit(self, instruction_text, task_info_dict, output_dir, max_turns,
                        harness_script_file):
        raise NotImplementedError
