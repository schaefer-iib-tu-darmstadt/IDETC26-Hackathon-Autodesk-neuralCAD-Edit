"""The stock CadQuery-script harness loop (`BaseVLM.visual_update_loop`) and its `prompt_history`
switch.

Two prompt layouts exist. `"full"` is the organisers' loop: every iteration's script, program
output and render are appended to one growing message list. `"last"` rebuilds the prompt each
iteration from the fixed instruction block plus only the previous iteration's feedback; this is the
variant the gemini-3.7-flash baseline numbers were produced with. Both are pinned here with a
model stub, so a change to either layout -- or to the default -- is caught before it silently
changes what "the baseline" means.
"""

import json
import os.path as osp

import pytest
from conftest import CONFIG_PATH

from src.vlms.base_vlm import BaseVLM, GenerateResponseResult


class RecordingVLM(BaseVLM):
    """Answers a fixed script every time and records the prompt it was given."""

    def __init__(self, config):
        super().__init__(config=config, cache=False)
        self.prompts = []

    def create_messages(self, inputs, sys=None):
        return list(inputs)

    def generate_response(self, messages, output_path=None, return_token_counts=False):
        self.prompts.append(list(messages))
        payload = {"complete": False, "my_cad_function": f"def my_cad_function(args): pass  # {len(self.prompts)}"}
        return GenerateResponseResult(
            response_json=payload, response_text=json.dumps(payload), thinking_text="",
            token_counts={"input_tokens": 10, "output_tokens": 5, "thinking_tokens": 0, "total_tokens": 15},
        )


def _run(tmp_path, config, iterations=3):
    harness = tmp_path / "harness.py"
    harness.write_text("# harness source\n", encoding="utf-8")
    vlm = RecordingVLM({"system_prompt": "sys", "1m_token_cost_input": 1.0,
                        "1m_token_cost_output": 2.0, **config})
    outputs = iter(f"stdout: run {i}\nstderr: " for i in range(1, iterations + 1))

    def run_function(script, harness_script_file, output_dir, input_file=None):
        return next(outputs)

    result = vlm.visual_update_loop(
        instruction_text="Add a hole.", task_info_dict={"request": "r1"},
        harness_script_file=str(harness), output_dir=str(tmp_path), max_iters=iterations,
        run_function=run_function, conversation_instruction="INSTRUCTION", input_file=None,
    )
    return vlm.prompts, result


def test_default_is_the_organisers_growing_history(tmp_path):
    prompts, result = _run(tmp_path, {})
    assert result["num_iterations"] == 3
    assert result["termination_reason"] == "max_iterations_reached"
    # Iteration 1: fixed block + first-iteration placeholder + remaining counter.
    assert prompts[0][0] == "INSTRUCTION" and prompts[0][1] == "# harness source\n"
    assert prompts[0][-2:] == ["Last output rendering: None (this is the first iteration)",
                               "Iterations remaining: 3"]
    # Iteration 3 still carries BOTH earlier rounds, in the organisers' order and with their labels.
    labels = [m for m in prompts[2] if isinstance(m, str) and m.endswith(":")]
    assert labels.count("Program output from last iteration:") == 2
    assert labels.count("The function that you produced from the last iteration:") == 2
    tail = prompts[2][-7:]
    assert tail[0] == "Program output from last iteration:"
    assert tail[1].startswith("stdout: run 2")
    assert tail[2] == "Visual output from last iteration:"
    assert tail[3] == "None"                       # no tmp.png was written by the stub
    assert tail[4] == "The function that you produced from the last iteration:"
    assert tail[5].endswith("# 2")
    assert tail[6] == "Iterations remaining: 1"
    assert len(prompts[2]) > len(prompts[1]) > len(prompts[0])


def test_last_mode_shows_only_the_previous_iteration(tmp_path):
    prompts, _ = _run(tmp_path, {"prompt_history": "last"})
    assert prompts[0][-2:] == ["Last output rendering: None (this is the first iteration)",
                               "Iterations remaining: 3"]
    # Iterations 2 and 3 have the same length: the feedback block is replaced, not appended.
    assert len(prompts[1]) == len(prompts[2])
    tail = prompts[2][-7:]
    assert tail[0] == "The function that you produced from the last iteration:"
    assert tail[1].endswith("# 2")
    assert tail[2] == "Program output from last iteration:"
    assert tail[3].startswith("stdout: run 2")
    assert tail[4] == "Visual output from last iteration:"
    assert tail[5] == "None"
    assert tail[6] == "Iterations remaining: 1"
    assert "stdout: run 1" not in json.dumps(prompts[2])


def test_last_mode_truncates_long_program_output_full_mode_does_not(tmp_path):
    long_output = "x" * 12000
    for mode, expect_truncated in (("last", True), ("full", False)):
        vlm = RecordingVLM({"system_prompt": "sys", "1m_token_cost_input": 1.0,
                            "1m_token_cost_output": 2.0, "prompt_history": mode})
        harness = tmp_path / "harness.py"
        harness.write_text("# harness\n", encoding="utf-8")
        vlm.visual_update_loop(
            instruction_text="i", task_info_dict={}, harness_script_file=str(harness),
            output_dir=str(tmp_path), max_iters=2,
            run_function=lambda *a, **k: long_output, conversation_instruction="c",
        )
        shown = vlm.prompts[1][vlm.prompts[1].index("Program output from last iteration:") + 1]
        assert ("[output truncated for prompt]" in shown) is expect_truncated, mode


def test_an_unknown_prompt_history_value_is_rejected_at_construction(tmp_path):
    """A typo must fail when the model is loaded, before the harness creates any task folder."""
    with pytest.raises(ValueError):
        RecordingVLM({"system_prompt": "sys", "prompt_history": "sometimes"})


def test_the_run_records_which_layout_produced_it(tmp_path):
    _, full = _run(tmp_path, {})
    _, last = _run(tmp_path, {"prompt_history": "last"})
    assert (full["prompt_history"], last["prompt_history"]) == ("full", "last")


def test_a_null_prompt_history_selects_the_default(tmp_path):
    """A copied config entry says `"prompt_history": null` to mean "no override"."""
    prompts, _ = _run(tmp_path, {"prompt_history": None})
    assert len(prompts[2]) > len(prompts[1])          # growing history, i.e. "full"


class ImageResolvingVLM(RecordingVLM):
    """Like the real `create_messages`: a path to an existing PNG is read at prompt-build time."""

    def create_messages(self, inputs, sys=None):
        out = []
        for message in inputs:
            if isinstance(message, str) and message.endswith(".png") and osp.exists(message):
                with open(message, "rb") as f:
                    out.append(("png", f.read()))
            else:
                out.append(message)
        return out


def _run_with_renders(tmp_path, config, iterations=3):
    harness = tmp_path / "harness.py"
    harness.write_text("# harness source\n", encoding="utf-8")
    vlm = ImageResolvingVLM({"system_prompt": "sys", "1m_token_cost_input": 1.0,
                             "1m_token_cost_output": 2.0, **config})
    calls = {"n": 0}

    def run_function(script, harness_script_file, output_dir, input_file=None):
        calls["n"] += 1
        with open(osp.join(output_dir, "tmp.png"), "wb") as f:
            f.write(f"render-{calls['n']}".encode())
        return f"stdout: run {calls['n']}\nstderr: "

    vlm.visual_update_loop(
        instruction_text="Add a hole.", task_info_dict={"request": "r1"},
        harness_script_file=str(harness), output_dir=str(tmp_path), max_iters=iterations,
        run_function=run_function, conversation_instruction="INSTRUCTION", input_file=None,
    )
    return vlm.prompts


def test_full_mode_shows_the_latest_render_in_every_history_slot(tmp_path):
    """The organisers' loop appends the same `tmp.png` path after every iteration and the path is
    re-read when the prompt is built, so by iteration 3 both history slots carry the render of
    iteration 2 -- the earlier render is gone. Pinned because it is what the shipped baselines
    (gemini-3-pro, gpt-5.2, claude-sonnet-4.5) were produced with."""
    prompts = _run_with_renders(tmp_path, {})
    images = [m for m in prompts[2] if isinstance(m, tuple)]
    assert len(images) == 2
    assert [b for _, b in images] == [b"render-2", b"render-2"]


def test_last_mode_shows_exactly_the_previous_render(tmp_path):
    prompts = _run_with_renders(tmp_path, {"prompt_history": "last"})
    images = [m for m in prompts[2] if isinstance(m, tuple)]
    assert images == [("png", b"render-2")]


def test_the_config_pins_the_baseline_variant():
    """The shipped baseline entry must say which loop produced its numbers."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    baseline = config["benchmark_models"]["gemini-3.7-flash_cadquery-script"]
    assert baseline["prompt_history"] == "last"
    assert baseline["function"] == "cadquery_script"
    # The organisers' own entries carry no override and therefore run the growing-history loop.
    assert "prompt_history" not in config["benchmark_models"]["gpt-5.2_cadquery-script"]
