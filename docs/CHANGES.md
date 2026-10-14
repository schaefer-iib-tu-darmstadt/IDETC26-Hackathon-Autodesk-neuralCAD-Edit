# What this fork changes

The base is the organisers' `main` at commit `bff9992`, and everything the agent needs is in this
branch. Every file not listed below is identical to the base; `git diff --name-status bff9992 HEAD`
regenerates the list. The metric code (`src/utils/evals_feature_geometric.py`,
`src/utils/evals_diff.py`, `src/utils/db.py`, `src/scripts/benchmark_evals/edit.py`) is
byte-identical to upstream, so every score is computed by the organisers' own implementation.

A = added, M = modified, both relative to `bff9992`.

## 1. Agent (new code)

| File | A/M | What it is |
|---|---|---|
| `src/agents/cad_agent_v4.py` | A | The agent: four tools (`ground`, `probe`, `apply_edit`, `finish`) on an ADK `LlmAgent`, one call budget (`max_turns`), the loop around them and the choice of the delivered checkpoint |
| `src/agents/cad_agent_base.py` | A | Tool-independent runner base: Gemini model object, generation config, the `run_edit` entry point, prompt-part helpers, transport-retry classifier, fixed between-turn messages |
| `src/agents/cad_graph.py` | A | B-Rep graph behind `ground`: face/edge/solid queries that return revision-scoped handles (`c3:F12`); a handle from an older revision is refused as `stale_handle` |
| `src/agents/cad_contract.py` | A | Delta contracts: normalises the EditSpec and the per-step expectation, runs the preflights before any script executes, scores each commit into a DeltaCard, picks the delivery commit |
| `src/agents/cad_git.py` | A | Git4CAD checkpoint history (`CadRepo`): commit, revert, HEAD; the commit gates (origin identity, validity relative to the input, preserved ratio, lost solids); per-commit face delta; final ranking; verbatim start-STL copy when the input is delivered |
| `src/agents/cad_tools.py` | A | SDK-free geometry layer: shape fingerprint (`summarise_shape`), change test (`_differs`), one-sided damage check (`assess_damage`), edit and probe subprocess runners, discovery of the seven dataset start renders |
| `src/agents/cad_trace.py` | A | ADK plugin that appends one JSON event per model and tool call to `run_trace.jsonl` in the run folder |
| `src/agents/cad_render.py` | A | Subprocess that renders a red/green diff of two STEP states; reached only with `features.diff_render`, which the shipped config switches off |
| `src/harnesses/cadquery_probe.py` | A | Read-only worker behind `probe`: runs `my_cad_function` and returns stdout, with no STEP export and no render |
| `src/vlms/gemini_auth.py` | A | One credential resolver for the stock harness and the agent: `GOOGLE_CLOUD_PROJECT` selects Vertex AI with ADC, else `GOOGLE_API_KEY` / `GEMINI_API_KEY`; loads only the repository-root `.env` |

## 2. Organisers' pipeline, modified

None of these edits reaches the metric code. They change how the model is called, how tasks are
scheduled and logged, and how `cadquery_convert.py` renders the seven PNG views used for ingestion;
no scored metric reads an image.

| File | A/M | What changed |
|---|---|---|
| `src/vlms/gemini.py` | M | Credentials through `gemini_auth` (upstream: `GEMINI_API_KEY` only); safety filters `BLOCK_NONE`; exponential-backoff retry on 429, timeout, 5xx and network errors (`max_retries`, default 5); configurable `max_output_tokens` and `timeout_seconds`; `role="user"` on every request (Vertex requires it); a missing, empty or unreadable image goes as text instead of raising; token counts tolerate a missing `usage_metadata`; explicit error on an empty response; new method `agentic_cadquery_v4`, the agent's entry point |
| `src/vlms/base_vlm.py` | M | `visual_update_loop` gains `prompt_history`: `full` (default) is the organisers' growing-context loop, `last` resends only the previous iteration's feedback and trims long output; per-iteration timings, `termination_reason` and the layout used land in `settings.json`; UTF-8 file writes |
| `src/scripts_benchmark_inference/run_harness.py` | M | `--workers N` runs tasks in a thread pool; `edit_id` carries a `uuid4` suffix, so tasks started in the same clock tick cannot share an output folder; progress logging |
| `src/harnesses/cadquery_script.py` | M | Function file read as UTF-8; `--skip_render` exports the STEP without the PNG (used by the agent's edit runner) |
| `src/scripts_preprocess/cadquery_convert.py` | M | Hands the whole Workplane to `render_to_png` instead of `.val()`, which kept only the first solid, so multi-solid parts render completely; new `--views-only` and `--no-views`; `--no-skip` now re-exports instead of reporting existing files as processed |
| `src/utils/cadquery_rendering.py` | M | `render_to_png` collapses a Workplane, Assembly, Shape or `TopoDS_Shape` to one shape before branching, on every platform. Windows (VTK) branch only: camera framed on the projected bounding box at 0.8 fill instead of a fixed distance aimed at the origin; `vis.show`'s default roll −35° and elevation −45° set to 0, so each view looks along its named direction; grey instead of gold |
| `src/utils/visualise_results.py` | M | Bar plots skip models without any score; `_score_or_zero` (a failed edit counts as 0.0) is unchanged |

## 3. Configuration

| File | A/M | What changed |
|---|---|---|
| `src/config/edit_192_external.json` | M | Two `benchmark_models` entries, both also added to `benchmark_eval_users.edit`: `gemini-3.7-flash_cadquery-script` (stock harness, `thinking_level: medium`, `prompt_history: last`) and `gemini-3.7-flash_adk-agent-v4` (the agent: `function: agentic_cadquery_v4`, `max_turns: 10`, `seed`, `features` block) |
| `pyproject.toml` | M | Dependencies `google-adk>=2.7.0` and `python-dotenv>=1.2.2`; dev group `pytest>=9`; pytest settings (repository root on `sys.path`, markers `slow` and `dataset`, `slow` deselected by default) |
| `uv.lock` | M | Re-locked: adds `google-adk` 2.7.0 with its dependencies and `pytest`; `opentelemetry-*` moves from 1.44 to 1.43 and `websockets` from 16.1 to 15.0 |
| `environment.yml` | M | The same three additions for the conda path, plus a note to run from the repository root with `PYTHONPATH=.` |
| `.gitignore` | M | Generic Python template cut to what this repository produces; adds `.env.*`, `.claude/`, `.vscode/`, `.idea/`, `information/`, `hackathon/`, `tmp/`, `outputs/`, `mongita_db.backup*`; the `data/**` rule is unchanged |
| `.gitattributes` | A | Line endings: `text=auto`, LF for `.sh`, CRLF for `.ps1` |

## 4. Tests

`uv run pytest` needs no API key; only `test_bbox_derivation.py` is deselected by default (`slow`),
and the tests marked `dataset` skip when `data/edit_192_external` is absent.

| File | A/M | What it covers |
|---|---|---|
| `tests/conftest.py` | A | Session fixtures: a CadQuery-built four-solid part and the states an agent produces from it (edit, second edit, dropped solids, component removal, global move); paths |
| `tests/fake_llm.py` | A | `ScriptedLlm`, an ADK `BaseLlm` that answers from a fixed script and records everything the model was shown |
| `tests/test_v4_offline.py` | A | The whole agent with `ScriptedLlm`: ground-edit-finish, failed contract and revert, nudges, ADK call limit, transport retry, internal errors; pins outcome, delivered checkpoint, files and prompts |
| `tests/test_v4_regressions.py` | A | The v4 config entry (budget, `features`, seed), runtime accounting, STEP export without render |
| `tests/test_cad_contract.py` | A | Intent inference, preflights, EditSpec normalisation, contract scoring, delivery choice |
| `tests/test_cad_graph.py` | A | Graph queries, malformed and stale handles, handles surviving an unchanged snapshot |
| `tests/test_cad_git.py` | A | The commit gates, each test named after the measured failure it guards against |
| `tests/test_commit_report.py` | A | What a commit reports to the agent: face delta against HEAD, preserved ratio, solid roster |
| `tests/test_git4cad_regressions.py` | A | Checkpoint-layer regressions: start-STL copy, concurrent commit numbering, renderer failure, missing or corrupt origin, tracer robustness |
| `tests/test_cad_probe.py` | A | The stdout-only probe worker |
| `tests/test_cad_render.py` | A | The diff-render subprocess end to end |
| `tests/test_stock_harness_loop.py` | A | Both `prompt_history` layouts of the stock loop, and the baseline entry's choice of `last` |
| `tests/test_bbox_derivation.py` | A | `slow`, `dataset`: the per-solid bounding box equals the exact one on every start model (~6 min) |

## 5. Documentation

| File | A/M | What it is |
|---|---|---|
| `README.md` | M | Fork section inserted after the organisers' title and figure: approach, results, quick start, links into `docs/`. The organisers' README follows under a new heading, its text unchanged except two directory names in "Repo structure", corrected to `src/scripts_benchmark_inference` and `src/scripts_groundtruth` |
| `docs/ARCHITECTURE.md` | A | The agent module by module: module map, one task step by step, tools, handles, contract, checkpoints, delivery, configuration, run outputs, offline verification, results, other models |
| `docs/REPRODUCING.md` | A | Dataset, credentials, run and scoring commands, pitfalls, the stock baseline |
| `docs/CHANGES.md` | A | This overview |
| `docs/img/v4_walkthrough.png` | A | Figure 1 of the README, a walk-through of the agent |

## 6. Tools and scripts

| File | A/M | What it is |
|---|---|---|
| `src/scripts/check_gemini_access.py` | A | Credentials smoke test: resolves the backend through `gemini_auth` and sends one short request to the configured model (`--list` also lists the visible models) |

## Size and reading order

Agent code is about 4,000 lines in 10 new files, the edits to the organisers' pipeline about
+510 / −100 lines in 7 files, the tests about 1,860 lines in 13 files, configuration about
+100 / −150 lines without `uv.lock` (the removals are the trimmed `.gitignore`), and the smoke test
about 70 lines.

1. [`README.md`](../README.md), fork section and figure 1: the approach and the results.
2. [`ARCHITECTURE.md` §0–§3](ARCHITECTURE.md#0-module-map): module map, mental model, stock harness
   versus agent, one task step by step.
3. [`ARCHITECTURE.md` §6–§7](ARCHITECTURE.md#6-the-contract-cad_contractpy): the contract and the checkpoint gates, which decide what the agent keeps.
4. [`tests/test_v4_offline.py`](../tests/test_v4_offline.py) with `fake_llm.py`: the agent's
   behaviour as executable scenarios, runnable without an API key.
5. [`REPRODUCING.md`](REPRODUCING.md), before the first real run.
