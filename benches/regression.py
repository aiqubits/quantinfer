"""Shared regression limits and paired real-model report checks."""
import math
import re
import statistics

LATENCY_RATIO_LIMIT = 1.35
MIN_THROUGHPUT_RATIO = 0.70
PEAK_RSS_RATIO_LIMIT = 1.10
METRICS = (
    "model_load_ms", "prefill_ms", "ttft_ms", "profile_decode_ms",
    "profile_decode_tokens_per_sec", "peak_rss_kib",
    "kv_cache_bytes", "kv_cache_capacity_bytes",
)
IDENTITY = (
    "model_id", "source_revision", "variant", "gguf_sha256", "cpu_model", "cpu_arch",
    "threads", "build_profile", "max_sequence_length", "prompt_tokens", "decode_tokens",
    "generated_tokens", "runtime_provider",
)


def read_report(path):
    report = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in report:
                raise ValueError(f"{path}: duplicate report field {key}")
            report[key] = value
    return report


def check_metrics(report, expected, name, *, timing):
    limits = {"peak_rss_kib": (0.0, PEAK_RSS_RATIO_LIMIT)}
    for key in ("kv_cache_bytes", "kv_cache_capacity_bytes"):
        actual, baseline = int(report[key]), int(expected[key])
        if actual <= 0 or baseline <= 0 or actual != baseline:
            raise ValueError(f"{name}: {key} must exactly match the positive integer baseline")
        print(f"{name} {key} exact={actual}")
    if timing:
        limits.update({key: (0.0, LATENCY_RATIO_LIMIT)
                       for key in ("prefill_ms", "ttft_ms", "profile_decode_ms")})
        limits["profile_decode_tokens_per_sec"] = (MIN_THROUGHPUT_RATIO, math.inf)
    failures = []
    for key, (minimum, maximum) in limits.items():
        actual, baseline = float(report[key]), float(expected[key])
        if not all(math.isfinite(value) and value > 0 for value in (actual, baseline)):
            raise ValueError(f"{name}: {key} must be finite and positive")
        ratio = actual / baseline
        print(f"{name} {key} ratio={ratio:.3f} allowed=[{minimum:.2f}, {maximum:.2f}]")
        if not minimum <= ratio <= maximum:
            failures.append(key)
    if failures:
        raise ValueError(f"{name} exceeded regression limits for {', '.join(failures)}")


def aggregate_reports(reports):
    if len(reports) != 3:
        raise ValueError("paired comparisons require three samples per executable")
    first = reports[0]
    for report in reports:
        for key in (*IDENTITY, "benchmark_code_commit", "benchmark_cli_sha256",
                    "kv_cache_bytes", "kv_cache_capacity_bytes"):
            if report[key] != first[key]:
                raise ValueError(f"sample identity differs for {key}")
    result = dict(first)
    for key in METRICS:
        if key in ("kv_cache_bytes", "kv_cache_capacity_bytes"):
            if int(first[key]) <= 0:
                raise ValueError(f"{key} must be positive")
            continue
        values = [float(report[key]) for report in reports]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError(f"{key} must be finite and positive in every sample")
        result[key] = str(statistics.median(values))
    result["sample_count"] = "3"
    result["aggregation"] = "per-metric-median"
    return result


def check_paired(current, reference, reference_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", reference_commit):
        raise ValueError("reference must be a full pinned commit SHA")
    if reference["benchmark_code_commit"] != reference_commit:
        raise ValueError("reference report does not match the pinned code commit")
    for report in (current, reference):
        if not re.fullmatch(r"[0-9a-f]{40}", report["benchmark_code_commit"]):
            raise ValueError("missing benchmark code identity")
        if not re.fullmatch(r"[0-9a-f]{64}", report["benchmark_cli_sha256"]):
            raise ValueError("missing benchmark executable hash")
        if report["sample_count"] != "3" or report["aggregation"] != "per-metric-median":
            raise ValueError("paired reports must contain three-sample medians")
    for key in IDENTITY:
        if current[key] != reference[key]:
            raise ValueError(f"paired benchmark identity differs for {key}")
    check_metrics(current, reference, f"paired {current['model_id']} {current['variant']}", timing=True)
