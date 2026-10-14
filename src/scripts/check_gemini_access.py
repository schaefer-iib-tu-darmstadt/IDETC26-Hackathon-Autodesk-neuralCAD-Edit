"""Credentials smoke test: can this machine reach the Gemini model a config entry names?

Exercises exactly the credential path the harness and the agent use (`src/vlms/gemini_auth.py`),
so a green result here means `run_harness.py` will get past authentication. One short request is
sent; nothing is written.

    uv run python src/scripts/check_gemini_access.py --config src/config/edit_192_external.json \
        --userId gemini-3.7-flash_adk-agent-v4

    uv run python src/scripts/check_gemini_access.py --list     # base models the backend offers
"""

import argparse
import sys

from google import genai

from src.utils.process_config import load_config
from src.vlms.gemini_auth import describe_backend, resolve_gemini_client_kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default="src/config/edit_192_external.json")
    parser.add_argument("--userId", default="gemini-3.7-flash_adk-agent-v4",
                        help="benchmark_models entry whose `model` is tried")
    parser.add_argument("--model", default=None, help="override the model name from the config")
    parser.add_argument("--list", action="store_true", help="also list the base models available")
    args = parser.parse_args(argv)

    model = args.model
    if model is None:
        config = load_config(args.config)
        try:
            model = config["benchmark_models"][args.userId]["model"]
        except KeyError:
            print(f"No benchmark_models entry {args.userId!r} in {args.config}", file=sys.stderr)
            return 2

    try:
        client_kwargs = resolve_gemini_client_kwargs(timeout_seconds=60)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(f"Backend: {describe_backend(client_kwargs)}")
    client = genai.Client(**client_kwargs)

    if args.list:
        names = []
        for entry in client.models.list():
            name = entry.name or ""
            names.append(name.removeprefix("publishers/google/models/").removeprefix("models/"))
        print(f"{len(names)} models visible:")
        for name in sorted(names):
            print("  ", name)

    print(f"Model: {model}")
    response = client.models.generate_content(
        model=model, contents="Reply with the single word OK."
    )
    usage = response.usage_metadata
    tokens = usage.total_token_count if usage else "unknown"
    print(f"Response: {(response.text or '').strip()!r} | {tokens} tokens")
    print("OK: credentials and model are usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
