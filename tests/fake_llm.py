"""A scripted stand-in for the Gemini model, so the whole agent can run under test without an API.

`ScriptedLlm` is a `google.adk.models.BaseLlm`; ADK treats it exactly like the real model: the
runner builds the request (system instruction, tool declarations, conversation so far), the fake
answers from a fixed script, and every function call it emits goes through ADK's tool dispatch
into the agent's real `ground` / `probe` / `apply_edit` / `finish` closures. Only
`CadAgentRunnerV4._model` is replaced.

A script is a list of turns. A turn is one of
    str                       -> a final text answer
    [(tool_name, args), ...]  -> one function call per pair, all in one model turn
    callable(responses)       -> computed from the previous turn's tool results
    Exception instance        -> raised instead of answered (transport failure)

Every request the fake receives is recorded in `seen`, so a test can assert on what the model was
shown: the tool surface, the generation config, the system prompt, the user messages between
turns and each tool's response dict.
"""

from google.adk.models._capabilities import LlmCapabilities
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types

# Fixed per call, so token totals and the cost estimate are deterministic.
USAGE = dict(prompt_token_count=1000, candidates_token_count=50, thoughts_token_count=20,
             cached_content_token_count=0, total_token_count=1070)


class ScriptedLlm(BaseLlm):
    model: str = "scripted-fake"
    script: list = []
    seen: list = []

    @property
    def capabilities(self) -> LlmCapabilities:
        return LlmCapabilities(output_schema_and_tools=False)

    async def generate_content_async(self, llm_request, stream=False):
        last = llm_request.contents[-1] if llm_request.contents else None
        responses = [p.function_response.response for p in (last.parts or [])
                     if p.function_response] if last else []
        texts = [p.text for p in (last.parts or []) if p.text] if last else []
        cfg = llm_request.config
        record = {
            "call": len(self.seen) + 1,
            "n_contents": len(llm_request.contents),
            "last_role": last.role if last else None,
            "last_text": texts,
            "responses": responses,
            "tools": sorted(llm_request.tools_dict),
            "model": llm_request.model,
            "labels": dict(cfg.labels or {}),
            "system_instruction": str(cfg.system_instruction),
            "generation": {
                "temperature": cfg.temperature,
                "max_output_tokens": cfg.max_output_tokens,
                "seed": cfg.seed,
                "thinking_level": str(getattr(cfg.thinking_config, "thinking_level", None)),
                "include_thoughts": getattr(cfg.thinking_config, "include_thoughts", None),
                "safety": [(str(s.category), str(s.threshold))
                           for s in (cfg.safety_settings or [])],
            },
        }
        if self.script and isinstance(self.script[0], Exception):
            record["raised"] = repr(self.script[0])
            self.seen.append(record)
            raise self.script.pop(0)
        self.seen.append(record)

        turn = self.script.pop(0) if self.script else "script exhausted"
        if callable(turn):
            turn = turn(responses)
        if isinstance(turn, str):
            parts = [types.Part(text=turn)]
        else:
            parts = [types.Part(function_call=types.FunctionCall(name=name, args=args))
                     for name, args in turn]
        yield LlmResponse(
            content=types.Content(role="model", parts=parts),
            usage_metadata=types.GenerateContentResponseUsageMetadata(**USAGE),
            finish_reason=types.FinishReason.STOP,
            partial=False,
            turn_complete=True,
        )
