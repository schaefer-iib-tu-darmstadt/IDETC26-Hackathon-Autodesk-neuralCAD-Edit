# Reproducing the runs

The [README](../README.md) quick start is the short path; this file carries every command, flag and caveat.
The data is the organisers' Autodesk myshare zip (CC BY-NC 4.0 per the `LICENSE` file it ships), not the `autodesk/neuralCAD-Edit` release on Hugging Face.

```bash
uv sync          # Python 3.12, CadQuery, google-adk, pytest
uv run pytest    # ~3 min, no API key, no dataset
```

All commands run from the repository root. `uv sync` installs the project in editable mode, so the
`src.*` imports resolve under `uv run`; an interpreter outside the project's virtual environment
needs `PYTHONPATH` set to the repository root. The suite drives the whole agent with a scripted
model ([ARCHITECTURE.md §11](ARCHITECTURE.md#11-verifying-the-code-without-an-api)); one test needs
a dataset STEP and skips without it, and the dataset-wide sweep is opt-in with `-m slow`.

## 1. Dataset

Download the [pre-computed database zip](https://myshare.autodesk.com/:u:/g/personal/daniele_grandi_autodesk_com/IQAT9e3wi76SQ6zqJX6uCAn_AQ-6SOrZbl44TW_GuRkJbXo?e=iDQ8CG)
and extract it so that `data/edit_192_external/parquets/` and `data/edit_192_external/mongita_db/`
exist. Another location works if `storage_dir.path` in `src/config/edit_192_external.json` points
there. The organisers warn that the hackathon data differs significantly from earlier releases,
which rules out the Hugging Face dump. Everything under `data/` except `data/README.md` is
git-ignored, so neither the dataset nor any run output can reach a commit. Then run once:

```bash
uv run python src/scripts/filter_dataset_text_only.py --config src/config/edit_192_external.json
```

It prunes the database to text-conditioned requests, deletes `videos/` and every brep or frame
file the pruned database no longer references (keep the zip), and writes the 48 requests from
`parquets/val_edit_all.parquet` to `parquets/val_edit_text.parquet`, the input of every run below.

## 2. Credentials

Put one of the two in a `.env` file in the repository root:

```
GEMINI_API_KEY=...                       # public Gemini API (GOOGLE_API_KEY works too)
# or
GOOGLE_CLOUD_PROJECT=my-project          # Vertex AI; run `gcloud auth application-default login` once
GOOGLE_CLOUD_LOCATION=global             # optional, default "global"
```

`src/vlms/gemini_auth.py` resolves them for the stock harness and the agent alike, first match
wins: `GOOGLE_CLOUD_PROJECT` selects Vertex AI with application-default credentials, otherwise
`GOOGLE_API_KEY` or `GEMINI_API_KEY` selects the public API. Vertex AI always goes through a
project: the SDK's `GOOGLE_GENAI_USE_VERTEXAI` flag and Vertex "Express mode" keys are deliberately
not honoured, so that flag cannot switch backends. Variables already set in the shell win over
`.env`, because python-dotenv does not override them: a `GOOGLE_CLOUD_PROJECT` exported in the
shell selects Vertex AI even when `.env` holds an API key. Only the `.env` in the repository root
is loaded, and git ignores it. Without any credentials, loading the model fails
before the first task starts.

Then check that the model named in the config answers:

```bash
uv run python src/scripts/check_gemini_access.py --userId gemini-3.7-flash_adk-agent-v4
```

The script sends one short request through the same credential path and prints the backend
actually chosen, the model and the reply; `--list` also lists the models the backend offers. The
SDK checks application-default credentials only on the first request, so this is the step that
catches a missing `gcloud` login, a wrong model name or a project without access to the model.

## 3. Run the agent

Start with one request in a separate output folder and read back how it ended:

```bash
uv run python src/scripts_benchmark_inference/run_harness.py \
    --config src/config/edit_192_external.json \
    --input data/edit_192_external/parquets/val_edit_text.parquet \
    --output_dir data/smoke_outputs \
    --harness src/harnesses/cadquery_script.py \
    --userId gemini-3.7-flash_adk-agent-v4 \
    --required-extensions step --n-rows 1
uv run python -c "import glob, json; [print(s.get('failed_run'), s.get('termination_reason'), s.get('final_source'), s.get('agent_error')) for p in glob.glob('data/smoke_outputs/**/settings.json', recursive=True) for s in [json.load(open(p))]]"
```

`True` in the first column flags a task that left no `tmp.step` or raised outside the agent; in
the latter case the other three fields are `None` and the exception went to the console only.
`error_before_commit` together with a non-empty `agent_error` usually means the model could not
be reached ([§5](#5-pitfalls)); any other termination reason means the pipeline works. A folder
outside `data/model_outputs` stays out of the ingest and out of the full run's resume list. The
full run on all 48 requests:

```bash
uv run python src/scripts_benchmark_inference/run_harness.py \
    --config src/config/edit_192_external.json \
    --input data/edit_192_external/parquets/val_edit_text.parquet \
    --output_dir data/model_outputs \
    --harness src/harnesses/cadquery_script.py \
    --userId gemini-3.7-flash_adk-agent-v4 \
    --required-extensions step --workers 4
```

On Windows PowerShell replace the trailing `\` with a backtick or put the command on one line
(`src/scripts/run_all.ps1` shows the PowerShell form). `--workers N` runs N tasks in parallel
threads, and `--n-rows N` caps a batch at the first N pending requests. Mind the spelling:
`--output_dir` and `--remove_failed` take underscores, `--n-rows` and `--required-extensions`
hyphens.

Each task lands in
`data/model_outputs/cadquery_script/<userId>/val_edit_text/<edit_id>/brep_end/<t>/`, named after
the harness file and the input parquet. The folder holds `tmp.step`, `settings.json` (every
counter, the contract verdicts, `termination_reason`, the tool-call log) and `iteration_output/`
(every attempt's script, stdout and STEP, every checkpoint, `run_trace.jsonl`);
[ARCHITECTURE.md §10](ARCHITECTURE.md#10-what-a-run-writes) lists the files.

**Resume and retry.** At start-up `run_harness.py` skips every request that already has a
`settings.json` under the output folder, whether it failed or not. `--remove_failed` deletes in
two places. During the batch it removes a task's folder at once when the call into the agent
raised. After the batch it removes every task folder that lacks `settings.json` or a file for
each `--required-extensions` entry (here a `.step`); it never reads the `failed_run` flag. To
retry those tasks, run once with `--remove_failed`, then run again. The agent itself catches model
and transport errors: it ends with `termination_reason` `error_before_commit`,
`error_after_commit` or, after `finish`, `agent_completed_after_transport_error`, still writes a
`tmp.step`, and `run_harness.py` records the task as successful. Neither cleanup touches such a
task; delete its `<edit_id>` folder by hand to re-queue it.

## 4. Score

```bash
uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs/cadquery_script/gemini-3.7-flash_adk-agent-v4
uv run python src/scripts/build_instructions_db.py --config src/config/edit_192_external.json
uv run python src/scripts/run_all_benchmarks.py --config src/config/edit_192_external.json
```

* **Conversion.** `cadquery_convert.py` writes the `.stl` and seven PNG views next to each
  delivered `tmp.step` and only produces files that are missing. Run it exactly like this:
  `--no-skip` alone re-exports and overwrites the dataset `.stl` the agent ships on unchanged
  tasks. `--views-only` never touches meshes and renders all seven views of every file
  that lacks at least one;
  `--views-only --no-skip` re-renders all of them and is the one safe use of `--no-skip`.
  `--no-views` skips the renders, which no scored metric reads.
* **Ingest.** `build_instructions_db.py` ingests every run under `models_dir.paths`
  (`data/model_outputs` in the shipped config) and copies each delivered file into
  `data/edit_192_external/breps/` as `<userId>_<end time>.<ext>`.
* **Evaluation.** `run_all_benchmarks.py` first deletes every non-latest edit per (request, user)
  from the local database, so back up `data/edit_192_external/mongita_db` before experimenting. It
  scores the users listed in the config's `benchmark_eval_users`. It also embeds the
  `toprightiso` render of every request and edit without a DINO feature yet, using the config's
  `dino_model` (`facebook/dinov2-base`, downloaded from Hugging Face on first use); no scored
  metric reads those features. Scores already in the database are recomputed only with
  `"recompute_metrics": true` in the config.
* **Results.** `data/edit_192_external/results/all_results.json`, with `metric_bar_facets.png`
  and `cost_barplot.png` next to it. The JSON nests per-request scores as difficulty
  (`edit_easy`, `edit_medium`, `edit_hard`) → user → metric → request id, and a request without a
  score is `null`. For the 48-task mean, average over all 48 requests and count `null` and NaN as
  0.0, as the plots do:

  ```bash
  uv run python -c "import json; r = json.load(open('data/edit_192_external/results/all_results.json')); u = 'gemini-3.7-flash_adk-agent-v4'; [print(m, len(v), round(sum((x or 0) if x == x else 0 for x in v) / len(v), 4)) for m in ('chamfer_similarity_norm', 'volume_f1', 'diff_f1') for v in [[x for d in r for x in r[d][u][m].values()]]]"
  ```

**Reading the numbers.**

* `thinking_level: medium` with a fixed `seed` is near-deterministic on about half of the requests
  and not on the rest; the agent forwards `seed`, the stock loop sends none. A single 48-task run
  therefore cannot resolve differences below the run-to-run spread given in
  [ARCHITECTURE.md §13](ARCHITECTURE.md#13-results), and a fresh run will not match the README
  table to the third decimal.
* All three metrics compare STL meshes in the input frame without registration (`pre_align=False`),
  so nothing undoes a translation, rotation or unit change of the part. It is scored as a shape
  difference, which usually takes Diff F1 and Volume F1 to near zero. Chamfer samples 10,000
  points per mesh without a fixed seed, so differences below about 0.004 are sampling noise.
* On three of the 48 requests the ground-truth edit is voxel-identical to the start. Diff F1 then
  returns 1.0 only when the prediction's voxel diff is empty as well. The dataset's own `.stl`
  achieves that on all three; a CadQuery re-export of the identical STEP scored 0.0 on the one
  request where this was measured. The agent copies the dataset `.stl` whenever it delivers the
  unmodified input, so the detail matters on those three requests and only when the agent leaves
  them unchanged ([ARCHITECTURE.md §8](ARCHITECTURE.md#8-delivery)).

## 5. Pitfalls

* **An unreachable model looks like a finished run.** With a wrong model name, project, location
  or missing permission, the first model call raises inside the agent. The agent records
  `termination_reason: error_before_commit` with the exception in `agent_error`, ships the
  unmodified input with the dataset `.stl` and returns normally. `run_harness.py` therefore sets no
  `failed_run` flag, the task counts as done, and a run that fails this way throughout scores like
  the no-op row of the README table. Smoke-test with `--n-rows 1` first ([§3](#3-run-the-agent))
  and check `failed_run`, `termination_reason` and `agent_error`; `check_gemini_access.py`
  ([§2](#2-credentials)) catches most causes before any run.
* **A second run needs its own userId.** Requests already present in the output folder are
  skipped, and the evaluator keeps only the latest edit per (request, user). To keep two runs side
  by side, copy the config entry under a new key, add that key to the `benchmark_eval_users.edit`
  list and pass it as `--userId`; the key names both the output folder and the database user.
* **Windows path length.** Keep the clone's absolute path under about 50 characters, for example
  `C:\src\neuralcad-edit`. A task's deepest files sit about 200 characters below the repository
  root, and not every library in the chain accepts a path of 260 or more: open3d 0.19 cannot open
  one even with `LongPathsEnabled` set. The evaluator itself reads the short-named copies in
  `data/edit_192_external/breps/`, so the risk lies with tools that open a run folder in place.
* **Moving the clone.** A plain `uv sync` keeps the editable install pointing at the old location,
  and `uv run python src/...` then fails to import `src` or imports the old copy. Run
  `uv sync --reinstall-package neuralcad-edit` after a move.

## 6. The stock baseline

The organisers' CadQuery-script loop runs through the same command with the other config entry:

```bash
uv run python src/scripts_benchmark_inference/run_harness.py \
    --config src/config/edit_192_external.json \
    --input data/edit_192_external/parquets/val_edit_text.parquet \
    --output_dir data/model_outputs \
    --harness src/harnesses/cadquery_script.py \
    --userId gemini-3.7-flash_cadquery-script \
    --required-extensions step --workers 4
uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs/cadquery_script/gemini-3.7-flash_cadquery-script
```

Ingest and evaluation are those of [§4](#4-score). The entry uses the agent's model and
`thinking_level: medium`, at most ten iterations per task, and no seed. Unlike the agent, the stock
loop lets a model error propagate to `run_harness.py`, which marks the task `failed_run: true` and,
under `--remove_failed`, deletes it at once. `prompt_history` selects the prompt layout. `"full"`,
the organisers' loop and the default when the key is absent or null, keeps every earlier
iteration's script and program output in the prompt, so the context grows each round. Its image
slots all name the same `tmp.png`, which is re-read when the prompt is built, so every slot shows
the latest render. `"last"` shows only the previous iteration's script, output and render on top
of the fixed instruction block and trims over-long program output. The shipped entry sets `"last"`
because the README's stock-baseline row was produced with that variant; delete the key or set
`"full"` to run the organisers' loop unchanged. Every task that completes records its layout as
`prompt_history` in `settings.json` (a task that raised records none), and
`tests/test_stock_harness_loop.py` pins both layouts, the shared render slot included.
