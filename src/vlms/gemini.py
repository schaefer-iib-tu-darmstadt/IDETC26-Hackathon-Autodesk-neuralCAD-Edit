# Note: Gemini/Vertex AI backend with retry handling, safety settings, and token tracking.
from .base_vlm import BaseVLM, GenerateResponseResult
from .gemini_auth import describe_backend, resolve_gemini_client_kwargs
import os
import time
from google import genai
import json

from google.genai import types


class VLM(BaseVLM):
    """
    A Vision-Language Model (VLM) implementation that interacts with the Gemini API.
    """

    def __init__(self, config: dict, cache: bool = True):
        """
        Initializes the GeminiVLM against either backend the google-genai SDK offers.

        The backend decision (Vertex AI + ADC when GOOGLE_CLOUD_PROJECT is set, else the public
        Gemini API with GEMINI_API_KEY / GOOGLE_API_KEY) lives in `src/vlms/gemini_auth.py`,
        shared with the ADK agent, so both paths read the same variables.
        """

        super().__init__(config=config, cache=cache)
        self.config = config

        self.timeout = float(self.config.get("timeout_seconds", 180.0))
        client_kwargs = resolve_gemini_client_kwargs(self.timeout)
        self.backend = describe_backend(client_kwargs)
        self.client = genai.Client(**client_kwargs)


    def create_messages(self, inputs, sys=None):
        """
        Creates a list of messages for the Gemini API.

        Args:
            inputs (list): List of input frames.
            sys (str, optional): System message. Defaults to None.

        Returns:
            list: List of messages formatted for the Gemini API.
        """

        parts = []
        for i, part in enumerate(inputs):
            part_str = str(part) if part is not None else "None"
            # Note: Handle image payloads and empty/corrupted renderings safely
            if isinstance(part, str) and part.endswith(".png") and os.path.exists(part):
                try:
                    image_bytes = open(part, "rb").read()
                    if len(image_bytes) > 0:
                        parts.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type='image/png')))
                    else:
                        parts.append(types.Part(text="Visual rendering: None (empty image file)"))
                except Exception as e:
                    parts.append(types.Part(text=f"Visual rendering: None (read error: {e})"))
            elif isinstance(part, str) and (part.endswith(".jpg") or part.endswith(".jpeg")) and os.path.exists(part):
                try:
                    image_bytes = open(part, "rb").read()
                    if len(image_bytes) > 0:
                        parts.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type='image/jpeg')))
                    else:
                        parts.append(types.Part(text="Visual rendering: None (empty image file)"))
                except Exception as e:
                    parts.append(types.Part(text=f"Visual rendering: None (read error: {e})"))
            else:
                parts.append(types.Part(text=part_str))
                
        # Note: Vertex AI requires explicit role="user"; Gemini defaults it
        contents = types.Content(role="user", parts=parts)
        return contents


    def generate_response(self, messages: str, output_path=None, return_token_counts=False) -> str:
        if self.cache and output_path is not None and os.path.exists(output_path):
            with open(output_path, "r") as f:
                full_response = None
                thinking_text = None
                response = f.read()
        else:

            # Note: Safety filters set to BLOCK_NONE to prevent false positives on CAD code
            generation_config_args = {
                "system_instruction": self.config["system_prompt"],
                "temperature": 0.0,
                "max_output_tokens": self.config.get("max_output_tokens", 20000),
                "safety_settings": [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            }
            if "thinking_level" in self.config:
                generation_config_args["thinking_config"] = types.ThinkingConfig(thinking_level=self.config["thinking_level"], include_thoughts=True)

            generation_config = types.GenerateContentConfig(**generation_config_args)

            # Note: Retry with exponential backoff for transient 429 quota and timeout errors
            max_retries = int(self.config.get("max_retries", 5))
            base_delay = 5.0
            max_delay = 60.0
            full_response = None

            num_parts = len(messages.parts) if hasattr(messages, 'parts') else 1
            print(f"[Gemini/Vertex] Preparing request with {num_parts} content parts (Timeout: {self.timeout}s)...", flush=True)

            for attempt in range(1, max_retries + 1):
                try:
                    t_call_start = time.perf_counter()
                    print(f"[Gemini/Vertex] Attempt {attempt}/{max_retries}: Request sent, waiting for model response...", flush=True)
                    full_response = self.client.models.generate_content(
                        model=self.config["model"],
                        config=generation_config,
                        contents=messages,
                    )
                    t_call_elapsed = time.perf_counter() - t_call_start
                    usage = getattr(full_response, "usage_metadata", None)
                    in_tok = getattr(usage, "prompt_token_count", "?")
                    out_tok = getattr(usage, "candidates_token_count", "?")
                    th_tok = getattr(usage, "thoughts_token_count", 0)
                    print(f"[Gemini/Vertex] Response received in {t_call_elapsed:.2f}s | Tokens: In={in_tok}, Out={out_tok}, Thinking={th_tok}", flush=True)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    code = getattr(e, "code", None)
                    is_rate_limit = (code == 429) or ("resource_exhausted" in err_str) or ("quota" in err_str) or ("429" in err_str)
                    is_timeout = ("timeout" in err_str) or ("timed out" in err_str) or ("deadline" in err_str)
                    is_network_drop = any(phrase in err_str for phrase in (
                        "disconnect", "peer closed", "chunked read", "connection", "remote",
                        "broken pipe", "reset", "eof", "protocol", "unavailable", "bad gateway",
                    )) or isinstance(e, (ConnectionError, TimeoutError, OSError))
                    is_transient = (
                        is_rate_limit or is_timeout or is_network_drop or code in [500, 502, 503, 504]
                    )

                    if attempt < max_retries and is_transient:
                        # Exponential backoff
                        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                        if is_rate_limit:
                            print(f"\n[Gemini/Vertex Rate Limit] 429 Quota/Rate limit reached! Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}... Details: {e}", flush=True)
                        elif is_timeout:
                            print(f"\n[Gemini/Vertex Timeout] Socket timed out after {self.timeout}s! Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}... Details: {e}", flush=True)
                        elif is_network_drop:
                            print(f"\n[Gemini/Vertex Network Drop] Retrying after {delay:.1f}s: {e}", flush=True)
                        else:
                            print(f"\n[Gemini/Vertex Error] Transient error ({code or type(e).__name__}). Waiting {delay:.1f}s before retry {attempt+1}/{max_retries}... Details: {e}", flush=True)
                        time.sleep(delay)
                    else:
                        print(f"\n[Gemini/Vertex Error] Call failed permanently after {attempt} attempts: {e}", flush=True)
                        raise

            try:
                thinking_text = [part for part in full_response.parts if part.thought]
            except Exception:
                pass

            response = full_response.text

            if response is None:
                # Note: Fail fast with diagnostic details if model returned no text part
                candidates = full_response.candidates or []
                finish_reason = candidates[0].finish_reason if candidates else None
                usage = full_response.usage_metadata
                raise RuntimeError(
                    f"Gemini returned no text (finish_reason={finish_reason}, "
                    f"prompt_feedback={full_response.prompt_feedback}, usage={usage})"
                )

            # get rid of ```json and '''
            if response.startswith("```json"):
                # remove first and last lines
                response = response.split("\n")[1:-1]
                response = "\n".join(response)

        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            pass

        if output_path is not None:
            if output_path.endswith(".json"):
                with open(output_path, "w") as f:
                    json.dump(response, f, indent=4)
            else:
                with open(output_path, "w") as f:
                    f.write(response)
        

        response_object = GenerateResponseResult(
            response_json=response,
            response_text=getattr(full_response, 'text', str(response)) if full_response else str(response),
            thinking_text=str(thinking_text) if thinking_text else ""
        )

        if return_token_counts:
            # Note: Track input, output, thinking, and total token usage
            usage = getattr(full_response, "usage_metadata", None) if full_response else None
            token_count_dict = {
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "thinking_tokens": getattr(usage, "thoughts_token_count", 0) or 0,
                "total_tokens": getattr(usage, "total_token_count", 0) or 0,
            }
            response_object.token_counts = token_count_dict

        return response_object

    def agentic_cadquery_v4(self, instruction_text: str, task_info_dict: dict, harness_script_file: str, output_dir: str, **kwargs) -> dict:
        """Run the grounded-delta-contract editing agent (config `function: agentic_cadquery_v4`).

        Dispatched by name from `run_harness.py` via `getattr(vlm, config["function"])`, so this
        method name is part of the config contract.
        """
        from src.agents.cad_agent_v4 import CadAgentRunnerV4

        agent_runner = CadAgentRunnerV4(config=self.config)
        return agent_runner.run_edit(
            instruction_text=instruction_text,
            task_info_dict=task_info_dict,
            output_dir=output_dir,
            harness_script_file=harness_script_file,
            max_turns=kwargs.get("max_turns", self.config.get("max_turns", 10)),
        )
