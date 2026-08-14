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

