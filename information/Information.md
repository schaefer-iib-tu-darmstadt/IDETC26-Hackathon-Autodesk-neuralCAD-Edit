# Hackathon Autodesk Problem 3

## Web-Pages

Hackathon Problem: https://event.asme.org/Events/media/library/resources/idetc-cie/IDETC-2026-Hackathon-Autodesk-Problem-Statement.pdf
Hackathon Repo: https://github.com/grndnl/IDETC26-Hackathon-Autodesk-neuralCAD-Edit
Paper: https://arxiv.org/abs/2604.16170

## Initial Brainstorming

- Building a harness for the benchmark.
- Using Google PHD Fund (Google Cloud + Vertex) so mostly gemini ecosystem
- Framework for Gemini API calls => Google Gen AI SDK https://googleapis.github.io/python-genai/
- Framework for Agent-Harness => Google Agent Development Kit: https://adk.dev/get-started/python/


## Hackathon-Task
- information\Autodesk hackathon introduction - idetc 2026.pdf
- information\Autodesk One-Page-Problem-Statement-CIE-Hackathon_2026.pdf

- building a harness to maximize performance on automatic metrics
    Possible Improvements:
    - Passing additional metadata about the CAD model and its current state at different times during the edit
    - Using multiple agents to perform and guide different parts of the edit process
    - Providing tools to the VLM to help it in parts of the editing process it struggles with
    - Focus on improving the token efficiency

## Hackathon Evaluation

### Deliverables

- Code as pull request
- Output CAD files (.step and .stl) in standard format (incl. in repo)
- Scores, computed by provided eval-code
- Final presentation als PDF im Repo, inkl. Methodenüberblick, Scores und 5 ausgewählten Beispielen
  auf einer Folie

### Evaluation criteria

- 40 % Quantitative Eval (48 text-conditioned edits, mit dem bereitgestellten Code berechnet)
- 40 % Qualitative Evaluation (Method Eval: Novelty, Cost efficiency, Scalability)
- 20 % Other incl. Presentation quality, structure and clarity. Future work and broader impact discussion

### No-op Baseline
48 requests:
  chamfer_similarity_norm   0.971
  volume_f1                 0.899
  diff_f1                   0.062

### Gemini Harness Baseline

# 1. Inferenz. Nicht DB nur Dateisystem.
uv run python src/scripts_benchmark_inference/run_harness.py --config src/config/edit_192_external.json --input data/edit_192_external/parquets/val_edit_text.parquet --output_dir data/model_outputs --harness src/harnesses/cadquery_script.py --userId gemini-3.7-flash_cadquery-script --required-extensions step

# 2. Konvertierung: STEP → STL + 7 Views.
uv run python src/scripts_preprocess/cadquery_convert.py data/model_outputs/cadquery_script/gemini-3.7-flash_cadquery-script

# 3. DB-Ingest.
uv run python src/scripts/build_instructions_db.py --config src/config/edit_192_external.json

# 4. Eval. Rechnet die drei Metriken für die neuen Edits, lässt die bestehenden unberührt (Key-Skip).
uv run python src/scripts/run_all_benchmarks.py --config src/config/edit_192_external.json