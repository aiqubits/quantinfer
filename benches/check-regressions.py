"""Compare the real-model reports with recorded CPU and memory baselines."""

import argparse
import json
from pathlib import Path

from regression import check_metrics, read_report


def check_report(report_path, baselines):
    report = read_report(report_path)
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
    same_cpu = [run for profile, run in profiles if profile["cpu_model"] == report["cpu_model"]]
    memory = same_cpu or [run for _, run in profiles]
    expected = dict(memory[0])
    expected["peak_rss_kib"] = max(run["peak_rss_kib"] for run in memory)
    check_metrics(report, expected, name, timing=bool(same_cpu))
    if not same_cpu:
        print(f"{name} timing comparison unavailable for CPU {report['cpu_model']}")


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
