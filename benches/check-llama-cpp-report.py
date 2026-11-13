"""Check a llama.cpp comparison report produced by llama-cpp-compare.py.

This is separate from check-regressions.py. That script reads key=value
real-model reports and every JSON file under benches/baselines/. A comparison
report has a different schema, so it stays under benches/llama-cpp/.
"""

import argparse
import json
import math
import sys
from pathlib import Path

PHASES = (
    "profile_decode_matmul_ms",
    "profile_decode_attention_ms",
    "profile_decode_sampling_ms",
    "profile_decode_other_ms",
)


def fail(message):
    raise SystemExit(message)


def check(report):
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        fail("report has no runs")
    expected = report.get("generated_tokens")
    if not isinstance(expected, int) or expected < 1:
        fail("generated_tokens must be a positive count")
    for run in runs:
        threads = run.get("threads")
        if not isinstance(threads, int) or threads < 1:
            fail(f"invalid threads={threads!r}")
        quantinfer = run["quantinfer"]
        rss = quantinfer.get("peak_rss_kib")
        if not isinstance(rss, (int, float)) or not math.isfinite(rss) or rss <= 0:
            fail(f"threads={threads}: QuantInfer peak_rss_kib is missing")
        llama_rss = run["llama_cpp"].get("peak_rss_kib")
        if not isinstance(llama_rss, (int, float)) or not math.isfinite(llama_rss) or llama_rss <= 0:
            fail(f"threads={threads}: llama.cpp peak_rss_kib is missing")
        tokens = quantinfer.get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens or len(tokens) > expected:
            fail(f"threads={threads}: generated_token_ids is missing")
        if not all(isinstance(token, int) for token in tokens):
            fail(f"threads={threads}: generated_token_ids must be integers")
        for key in PHASES:
            value = quantinfer.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                fail(f"threads={threads}: {key} is missing")
        if quantinfer["profile_decode_matmul_ms"] <= 0:
            fail(f"threads={threads}: matmul time was not recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    check(json.loads(args.report.read_text()))
    print(f"{args.report}: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
