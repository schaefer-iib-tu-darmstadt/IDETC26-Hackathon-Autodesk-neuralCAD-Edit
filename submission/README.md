# IDETC 2026 Hackathon — Problem 3 Autodesk

**Team `human-in-the-loop`.** Final CAD output for the 48 text-conditioned edit requests, produced
by the Grounded Delta CAD Agent ("v4") running Gemini 3.7 Flash at medium thinking level.

- `cad/<request_id>/<request_id>.step` and `.stl`. 
- `scores.json`: per-task and aggregate `chamfer_similarity_norm`, `volumetric_f1` and `diff_f1`,
  as produced by `src/scripts/run_all_benchmarks.py`.
- Final presentation: `Presentation-Team-human-in-the-loop.pdf`.

The agent code is in `src/agents/` (`cad_agent_v4.py`, `cad_graph.py`, `cad_contract.py`,
`cad_git.py`, `cad_tools.py`) with user id `gemini-3.7-flash_adk-agent-v4`.

## Scores

| | Chamfer | Volume F1 | **Diff F1** |
|---|---:|---:|---:|
| **v4 Grounded Delta CAD Agent** | 0.9779 | 0.9176 | **0.4841** |
| Baseline CadQuery-script harness, same model and thinking level | 0.9740 | 0.9028 | 0.2824 |
| Second human expert (ceiling) | 0.9799 | 0.9058 | 0.5953 |


## Reproducing the scores

**Setup** — once:

```bash
uv sync
```

Download the dataset zip linked in the repository README and extract it to `data/edit_192_external`. It supplies the ground
truth edits our output is scored against.

### Compuate & Evaluate Results

Ingests the submitted files into the database as a separate user and runs the provided evaluator.

```bash
# 1. Lay the flat submission tree out in the shape build_instructions_db.py expects,
#    under the separate user id "submission-roundtrip". Writes a derived config.
uv run python src/scripts/stage_submission_for_db.py \
    --config src/config/edit_192_external.json --copy-stl

# 2. Ingest
uv run python src/scripts/build_instructions_db.py \
    --config src/config/edit_192_external_submission_roundtrip.json

# 3. Score — writes results/all_results.json and metric_bar_facets.png under storage_dir
uv run python src/scripts/run_all_benchmarks.py \
    --config src/config/edit_192_external_submission_roundtrip.json
```


Alternatively you can run this script:
```bash
uv run python src/scripts/verify_submission.py --config src/config/edit_192_external.json
```