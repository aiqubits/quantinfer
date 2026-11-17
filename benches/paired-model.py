"""Compare pinned/current executables on the same host, model and configuration."""
import argparse
import os
from pathlib import Path
import re
import subprocess

from regression import aggregate_reports, check_paired, read_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("current-cli", "reference-cli", "model", "output-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("current-commit", "reference-commit", "model-id", "source-revision", "variant"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    for commit in (args.current_commit, args.reference_commit):
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            parser.error("both code commits must be full SHAs")
    runner = Path(__file__).resolve().parent / "real-model.sh"
    samples = args.output_prefix.with_name(args.output_prefix.name + "-samples")
    samples.mkdir(parents=True, exist_ok=False)
    reports = {"current": [], "reference": []}
    for sample in range(3):
        # Alternate first position to reduce systematic ordering/cache bias.
        order = ("reference", "current") if sample % 2 == 0 else ("current", "reference")
        for kind in order:
            path = samples / f"{kind}-{sample + 1}.txt"
            environment = dict(os.environ)
            environment["QUANTINFER_BENCH_CODE_COMMIT"] = getattr(args, kind + "_commit")
            with path.open("w") as output:
                subprocess.run(
                    ["bash", str(runner), str(getattr(args, kind + "_cli").resolve()),
                     args.model_id, args.source_revision, args.variant, str(args.model.resolve())],
                    env=environment, stdout=output, check=True, timeout=180,
                )
            reports[kind].append(read_report(path))
    medians = {kind: aggregate_reports(values) for kind, values in reports.items()}
    for kind, report in medians.items():
        report["cache_state"] = "uncontrolled"
        path = args.output_prefix.with_name(args.output_prefix.name + f"-{kind}.txt")
        path.write_text("".join(f"{key}={value}\n" for key, value in report.items()))
    check_paired(medians["current"], medians["reference"], args.reference_commit)


if __name__ == "__main__":
    main()
