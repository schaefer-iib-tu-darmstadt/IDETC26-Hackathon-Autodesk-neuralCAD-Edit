"""Run tracing for the CAD agents -- one JSONL file per edit request.

`settings.json` records the *outcome* of a run: totals, the git log, the exported candidate. It
cannot answer why a run went the way it did, because it only sees what the agent chose to keep.
Establishing that a single turn had cost 118 s in the diff renderer took three ad-hoc scripts
after the fact, and the answer only came from timing the renderer by hand.

This plugin closes that gap by hanging off ADK's own callback surface (`BasePlugin`, 15 hooks in
2.7.0) and writing one JSON object per event, appended as it happens:

    {"t": 12.44, "ev": "tool.end",   "tool": "run_script", "sec": 96.1, "ok": true, ...}
    {"t": 108.9, "ev": "model.end",  "in": 63120, "out": 372, "think": 592, "sec": 11.8}
    {"t": 109.0, "ev": "tool.error", "tool": "render_view", "error": "..."}

Why JSONL and not a hosted tracer: the harness runs tasks through a `ThreadPoolExecutor`, so a
single global collector would interleave events from concurrent requests. A file per request is
inherently thread-safe, survives a crashed run, needs no server, and stays in the output folder
next to the artefacts it describes. ADK's OpenTelemetry spans remain available on top of this --
`google.adk.telemetry.tracer` emits `execute_tool <name>` spans regardless of this plugin, so
attaching an OTLP exporter later needs no change here.

Timings are wall-clock and measured around the callback pair, so `tool.end.sec` includes the
CadQuery subprocess and the diff render -- which is exactly the number that was missing.
"""

import json
import os
import os.path as osp
import threading
import time
from typing import Any, Optional

try:
    from google.adk.plugins import BasePlugin
except ImportError:  # keep this module importable without the SDK, like cad_git/cad_tools
    BasePlugin = object


def _usage(llm_response):
    """(input, output, thinking) from an LlmResponse, tolerating absent fields."""
    u = getattr(llm_response, "usage_metadata", None)
    if not u:
        return 0, 0, 0
    return (getattr(u, "prompt_token_count", 0) or 0,
            getattr(u, "candidates_token_count", 0) or 0,
            getattr(u, "thoughts_token_count", 0) or 0)


# Generous but bounded: a thought summary runs to a few hundred words, and the trace is read by
# humans and by analysis scripts, never sent back to the model.
THOUGHT_CHARS = 4000
SAY_CHARS = 1500


def _split_text(llm_response):
    """(visible_text, thought_text) from one LLM response. Empty strings when absent."""
    said, thought = [], []
    try:
        parts = (llm_response.content.parts if llm_response.content else None) or []
    except AttributeError:
        return "", ""
    for part in parts:
        text = getattr(part, "text", None)
        if not text:
            continue
        (thought if getattr(part, "thought", False) else said).append(text)
    return " ".join(said).strip(), "\n".join(thought).strip()


class CadRunTracer(BasePlugin):
    """Append-only JSONL trace of one agent run.

    Instances are per-request, so no cross-task locking is needed; the lock only guards the file
    against ADK's own async interleaving within a single run.
    """

    def __init__(self, path, name="cad_run_tracer", extra=None):
        try:
            super().__init__(name=name)
        except TypeError:                      # BasePlugin fallback stub
            pass
        self.path = path
        self.t0 = time.time()
        self._lock = threading.Lock()
        # Keyed on ADK's function_call_id, not id(tool_context): a cancelled sibling call (ADK
        # cancels the others when one parallel call raises) leaves its entry behind, and a later
        # context allocated at the same address would then pop a stale start time and report an
        # inflated duration straight into settings.json.
        self._tool_started: dict[str, float] = {}
        self._model_started: Optional[float] = None
        self.counts = {"model_calls": 0, "tool_calls": 0, "tool_errors": 0, "model_errors": 0}
        self.tool_seconds: dict[str, float] = {}
        # Best-effort from here on. Losing the trace costs a diagnostic; raising would cost the
        # whole task, and the harness path is long enough to hit Windows' MAX_PATH.
        self.disabled = False
        self.error = None
        try:
            os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"t": 0.0, "ev": "run.start", **(extra or {})}) + "\n")
        except OSError as e:
            self.disabled = True
            self.error = f"{type(e).__name__}: {e}"
            print(f"[cad-trace] disabled, could not open {path}: {e}", flush=True)

    # -- writing -------------------------------------------------------------------------------

    def _emit(self, ev, **fields):
        if self.disabled:
            return
        rec = {"t": round(time.time() - self.t0, 3), "ev": ev, **fields}
        line = json.dumps(rec, default=str)
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except OSError:
            self.disabled = True

    def summary(self):
        """Aggregate for `settings.json`, so the headline numbers need no log parsing."""
        out = {
            **self.counts,
            "wall_seconds": round(time.time() - self.t0, 1),
            "seconds_by_tool": {k: round(v, 1) for k, v in sorted(self.tool_seconds.items())},
            "trace_file": None if self.disabled else self.path,
        }
        if self.disabled:
            out["trace_error"] = self.error or "tracing was disabled"
        return out

    # -- ADK hooks -----------------------------------------------------------------------------
    # All hooks return None: this plugin observes, it never intercepts. Returning a value from
    # before_tool_callback would REPLACE the tool result and silently skip execution.

    @staticmethod
    def _call_key(tool_context):
        return getattr(tool_context, "function_call_id", None) or f"addr:{id(tool_context)}"

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self._tool_started[self._call_key(tool_context)] = time.time()
        args = {k: (f"<{len(v)} chars>" if isinstance(v, str) and len(v) > 200 else v)
                for k, v in (tool_args or {}).items()}
        self._emit("tool.start", tool=getattr(tool, "name", str(tool)), args=args)
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        name = getattr(tool, "name", str(tool))
        started = self._tool_started.pop(self._call_key(tool_context), None)
        sec = round(time.time() - started, 2) if started else None
        if sec is not None:
            self.tool_seconds[name] = self.tool_seconds.get(name, 0.0) + sec
        self.counts["tool_calls"] += 1
        fields: dict[str, Any] = {"tool": name, "sec": sec}
        if isinstance(result, dict):
            # Only the decision-carrying keys -- the full result holds image parts and stdout.
            for k in ("committed", "commit", "duplicate_of", "head_advanced", "is_valid",
                      "validity_regressed", "damaged", "preserved_ratio", "ok", "reason"):
                if k in result:
                    fields[k] = (result[k][:200] if isinstance(result[k], str) else result[k])
            fields["images"] = sum(1 for k in result if k.startswith("diff_") or k.startswith("view_"))
        self._emit("tool.end", **fields)
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        name = getattr(tool, "name", str(tool))
        self.counts["tool_errors"] += 1
        started = self._tool_started.pop(self._call_key(tool_context), None)
        # Counted like a success: a tool that spends its whole timeout and then throws is exactly
        # the one whose cost belongs in `seconds_by_tool`. Dropping it made the total look healthy.
        sec = round(time.time() - started, 2) if started else None
        if sec is not None:
            self.tool_seconds[name] = self.tool_seconds.get(name, 0.0) + sec
        self._emit("tool.error", tool=name, sec=sec, error=str(error)[:500])
        return None

    async def before_model_callback(self, *, callback_context, llm_request):
        self._model_started = time.time()
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        started = self._model_started
        self.counts["model_calls"] += 1
        tin, tout, tthink = _usage(llm_response)
        fields = {"in": tin, "out": tout, "think": tthink}
        # `thinking_config(include_thoughts=True)` is set, so thought summaries arrive on every
        # response and are billed -- 12 329 thinking tokens on one measured task. The agent loop
        # filters them out of `last_text` and nothing else looked at them, so the only record of
        # how the model reasoned was whatever it happened to write into script comments. Keeping
        # them costs one dict lookup and makes a finished run readable.
        said, thought = _split_text(llm_response)
        if thought:
            fields["thought"] = thought[:THOUGHT_CHARS]
        if said:
            fields["say"] = said[:SAY_CHARS]
        self._emit("model.end", sec=round(time.time() - started, 2) if started else None, **fields)
        self._model_started = None
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        self.counts["model_errors"] += 1
        self._emit("model.error", error=str(error)[:500])
        return None

    async def after_run_callback(self, *, invocation_context):
        self._emit("run.end", **self.counts)
        return None
