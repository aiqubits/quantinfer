"""Regression checks must detect slowdowns and mismatched benchmark identities."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from regression import METRICS, aggregate_reports, check_paired, read_report


def sample():
    report = dict.fromkeys(METRICS, "100")
    report.update({
        "model_id": "fixture", "source_revision": "pinned-model", "variant": "f32",
        "gguf_sha256": "b" * 64, "cpu_model": "test CPU", "cpu_arch": "x86_64",
        "threads": "1", "build_profile": "release", "max_sequence_length": "64",
        "prompt_tokens": "[1, 2, 3, 4, 5]", "decode_tokens": "8",
        "generated_tokens": "[1, 2, 3, 4, 5, 6, 7, 8]", "runtime_provider": "cpu-scalar",
        "benchmark_code_commit": "a" * 40, "benchmark_cli_sha256": "b" * 64,
    })
    return report


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.reference = aggregate_reports([sample() for _ in range(3)])

    def check(self, current):
        with contextlib.redirect_stdout(io.StringIO()):
            check_paired(current, self.reference, "a" * 40)

    def test_identical_reports_pass_and_each_regression_fails(self):
        self.check(copy.deepcopy(self.reference))
        for key, value in (("prefill_ms", "136"), ("ttft_ms", "136"),
                           ("profile_decode_ms", "136"), ("profile_decode_tokens_per_sec", "69"),
                           ("peak_rss_kib", "111"), ("kv_cache_bytes", "101")):
            with self.subTest(key=key):
                current = dict(self.reference, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    self.check(current)

    def test_cpu_model_artifact_configuration_and_code_identity_are_required(self):
        for key in ("cpu_model", "gguf_sha256", "threads", "prompt_tokens", "generated_tokens",
                    "runtime_provider", "benchmark_code_commit", "benchmark_cli_sha256", "sample_count"):
            with self.subTest(key=key):
                current = dict(self.reference, **{key: "invalid"})
                with self.assertRaises(ValueError):
                    self.check(current)
        with self.assertRaisesRegex(ValueError, "pinned code"):
            check_paired(self.reference, self.reference, "c" * 40)

    def test_median_limits_outlier_and_invalid_or_changed_samples_fail(self):
        samples = [sample() for _ in range(3)]
        samples[0]["prefill_ms"] = "1000"
        self.assertEqual(aggregate_reports(samples)["prefill_ms"], "100.0")
        for invalid in ("nan", "inf", "0", "-1"):
            samples[0]["prefill_ms"] = invalid
            with self.assertRaises(ValueError):
                aggregate_reports(samples)
        samples = [sample() for _ in range(3)]
        samples[0]["benchmark_cli_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "identity differs"):
            aggregate_reports(samples)

    def test_kv_integer_comparison_does_not_round_differences_above_f64_precision(self):
        self.reference["kv_cache_bytes"] = str(2**53)
        current = dict(self.reference, kv_cache_bytes=str(2**53 + 1))
        with self.assertRaisesRegex(ValueError, "kv_cache_bytes"):
            self.check(current)

    def test_paired_driver_writes_six_raw_reports_and_gates_the_medians(self):
        # Synthetic executable exercises orchestration, not real-model accuracy.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"fixture")
            cli = root / "cli"
            metrics = {key: "100" for key in METRICS}
            metrics.update({"decode_tokens": "8", "max_sequence_length": "64",
                            "prompt_tokens": "[1, 2, 3, 4, 5]", "generated_tokens": "[1, 2, 3, 4, 5, 6, 7, 8]"})
            output = "\n".join(f"{key}={value}" for key, value in metrics.items())
            cli.write_text("#!/usr/bin/env python3\nprint(" + repr(output) + ")\n")
            cli.chmod(0o700)
            command = [sys.executable, str(Path(__file__).with_name("paired-model.py")),
                       "--current-cli", str(cli), "--reference-cli", str(cli),
                       "--current-commit", "c" * 40, "--reference-commit", "a" * 40,
                       "--model-id", "fixture", "--source-revision", "fixed", "--variant", "f32",
                       "--model", str(model), "--output-prefix", str(root / "paired")]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(len(list((root / "paired-samples").glob("*.txt"))), 6)
            current = read_report(root / "paired-current.txt")
            reference = read_report(root / "paired-reference.txt")
            self.assertEqual(current["sample_count"], "3")
            self.assertEqual(reference["benchmark_code_commit"], "a" * 40)
            self.assertEqual(current["benchmark_code_commit"], "c" * 40)


if __name__ == "__main__":
    unittest.main()
