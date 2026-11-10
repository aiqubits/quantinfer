"""Compare the real-model reports with recorded CPU and memory baselines."""

import argparse
import json
import math
from pathlib import Path

LATENCY_RATIO_LIMIT = 1.35
MIN_THROUGHPUT_RATIO = 0.70
PEAK_RSS_RATIO_LIMIT = 1.10


def check_report(report_path, baselines):
    report = dict(
        line.split("=", 1) for line in report_path.read_text().splitlines() if "=" in line
    )
    identity = ("model_id", "source_revision", "variant")
    profiles = [
        (baseline, run)
        for baseline in baselines
        if baseline["cpu_arch"] == report["cpu_arch"]
        and str(baseline["threads"]) == report["threads"]
        and baseline["build_profile"] == report["build_profile"]
        and str(baseline["max_sequence_length"]) == report["max_sequence_length"]
        and baseline["prompt_tokens"] == json.loads(report["prompt_tokens"])
        and str(baseline["generated_token_count"]) == report["decode_tokens"]
        for run in baseline["runs"]
        if all(run[key] == report[key] for key in identity)
    ]
    if not profiles:
        raise ValueError(f"{report_path}: missing baseline for this model and benchmark configuration")

    name = f"{report['model_id']} {report['variant']}"
    failures = []

    def gate(key, expected, minimum, maximum):
        ratio = float(report[key]) / expected
        print(f"{name} {key} ratio={ratio:.3f} allowed=[{minimum:.2f}, {maximum:.2f}]")
        if not math.isfinite(ratio) or not minimum <= ratio <= maximum:
            failures.append(key)

    same_cpu = [run for profile, run in profiles if profile["cpu_model"] == report["cpu_model"]]
    memory = same_cpu or [run for _, run in profiles]
    gate("peak_rss_kib", max(run["peak_rss_kib"] for run in memory), 0.0, PEAK_RSS_RATIO_LIMIT)
    for key in ("kv_cache_bytes", "kv_cache_capacity_bytes"):
        gate(key, memory[0][key], 1.0, 1.0)

    if same_cpu:
        for key in ("prefill_ms", "ttft_ms", "profile_decode_ms"):
            gate(key, same_cpu[0][key], 0.0, LATENCY_RATIO_LIMIT)
        gate("profile_decode_tokens_per_sec", same_cpu[0]["profile_decode_tokens_per_sec"],
             MIN_THROUGHPUT_RATIO, math.inf)
    else:
        print(f"{name} timing comparison unavailable for CPU {report['cpu_model']}")

    if failures:
        raise ValueError(f"{name} exceeded regression limits for {', '.join(failures)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / "baselines"
    baselines = [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]
    for report in args.reports:
        check_report(report, baselines)


if __name__ == "__main__":
    main()
