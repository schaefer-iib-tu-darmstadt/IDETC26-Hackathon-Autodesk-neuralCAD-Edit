#!/usr/bin/env python3
"""Read-only CadQuery probe worker.

The edit harness always exports STEP and renders an image. A geometry question should not pay
for either operation, so this worker executes ``my_cad_function(args)`` and reports stdout only.
The return value is deliberately discarded.
"""

import os
import re
import sys
import textwrap

# Launched as a script by `cad_tools.run_probe_script`, so the repo root is not on sys.path.
# Import the sibling edit harness by file location instead of relying on an editable install.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadquery_script import load_and_execute_function, parse_cadquery_args  # noqa: E402


def main(argv=None):
    args = parse_cadquery_args(
        list(sys.argv if argv is None else argv),
        required_args=["--output_dir", "--input_file", "--function_file"],
    )
    output_dir = os.path.expanduser(args.get("output_dir", "."))
    os.makedirs(output_dir, exist_ok=True)

    function_file = args.get("function_file")
    if not function_file:
        print("No function file provided")
        return 2
    function_file = os.path.expanduser(function_file)
    if not os.path.exists(function_file):
        print(f"Function file not found: {function_file}")
        return 2

    with open(function_file, "r", encoding="utf-8") as stream:
        function_string = stream.read()
    if not re.search(r"^\s*def\s+my_cad_function\s*\(", function_string, re.MULTILINE):
        function_string = (
            "def my_cad_function(args):\n"
            + textwrap.indent(function_string, "    ")
        )
        print("probe_auto_wrapped=1")
    print(f"Loading probe function from: {function_file}")
    result = load_and_execute_function(function_string, args)
    print(f"probe_returned_model={int(result is not None)}")
    print("Probe completed without exporting geometry or rendering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
