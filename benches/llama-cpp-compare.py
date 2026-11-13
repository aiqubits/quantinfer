"""Compare QuantInfer with a pinned llama.cpp build on the same GGUF.

QuantInfer runs the real-model convention (five fixed prompt tokens, greedy
decoding, 64-token context; see benches/real-model.sh). llama.cpp runs
llama-bench with the same prompt/generation lengths and thread counts. The
prompt contents differ (llama-bench synthesizes its own), so this is a
throughput comparison, not a token-equivalence check; use
tests/conformance/gguf_interop.py for token equivalence.

Example (llama.cpp revision pinned by .github/workflows/gguf-interop.yml):

    python benches/llama-cpp-compare.py \
      --cli target/release/quantinfer-cli \
      --llama-bench /path/to/llama-bench \
      --model /path/to/model.gguf \
      --threads 1,2,4,8 \
      --output benches/llama-cpp/report.json
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

PROMPT_TOKENS = "1,2,3,4,5"


def cpu_model():
    if platform.system() == "Linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or "unknown"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantinfer_run(cli, model, threads, generated):
    output = subprocess.check_output(
        [
            cli, "generate", "--input", model,
            "--tokens", PROMPT_TOKENS,
            "--max-new-tokens", str(generated),
            "--max-sequence-length", "64",
            "--temperature", "0",
            "--threads", str(threads),
            "--profile",
        ],
        text=True,
    )
    metrics = {
        key: value
        for key, value in (
            line.split("=", 1) for line in output.splitlines() if "=" in line
        )
    }
    expected = {
        "threads": str(threads),
        "decode_tokens": str(generated),
        "max_sequence_length": "64",
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise RuntimeError(f"unexpected {key}={metrics.get(key)!r}, want {value!r}")
    return {
        "model_load_ms": float(metrics["model_load_ms"]),
        "prefill_ms": float(metrics["prefill_ms"]),
        "ttft_ms": float(metrics["ttft_ms"]),
        "decode_ms": float(metrics["decode_ms"]),
        "profile_decode_tokens_per_sec": float(metrics["profile_decode_tokens_per_sec"]),
        "peak_rss_kib": (
            float(metrics["peak_rss_kib"])
            if metrics.get("peak_rss_kib", "").replace(".", "", 1).isdigit()
            else None
        ),
        "kv_cache_bytes": int(metrics["kv_cache_bytes"]),
        "kv_cache_capacity_bytes": int(metrics["kv_cache_capacity_bytes"]),
        "generated_token_ids": json.loads(metrics["generated_tokens"]),
        "profile_decode_matmul_ms": float(metrics["profile_decode_matmul_ms"]),
        "profile_decode_attention_ms": float(metrics["profile_decode_attention_ms"]),
        "profile_decode_sampling_ms": float(metrics["profile_decode_sampling_ms"]),
        "profile_decode_other_ms": float(metrics["profile_decode_other_ms"]),
    }


def resident_kib(stderr):
    for line in stderr.splitlines():
        if "maximum resident set size" in line:
            return int(line.split()[0]) / 1024
        if line.strip().startswith("Maximum resident set size"):
            return float(line.split()[-1])
    return None


def llama_run(llama_bench, model, threads, prompt_tokens, generated, repetitions):
    command = [
        llama_bench, "-m", model,
        "-p", str(prompt_tokens), "-n", str(generated),
        "-t", str(threads), "-r", str(repetitions),
        "-ngl", "0",
        "--no-warmup", "-o", "json",
    ]
    if platform.system() == "Darwin":
        command = ["/usr/bin/time", "-l", *command]
    elif platform.system() == "Linux":
        command = ["/usr/bin/time", "-v", *command]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    output = completed.stdout
    peak_rss_kib = resident_kib(completed.stderr)
    results = json.loads(output[output.index("["):])
    by_kind = {}
    for entry in results:
        if entry["n_prompt"] > 0 and entry["n_gen"] == 0:
            by_kind["prefill_tokens_per_sec"] = entry["avg_ts"]
        elif entry["n_gen"] > 0 and entry["n_prompt"] == 0:
            by_kind["decode_tokens_per_sec"] = entry["avg_ts"]
    missing = {"prefill_tokens_per_sec", "decode_tokens_per_sec"} - by_kind.keys()
    if missing:
        raise RuntimeError(f"llama-bench reported no {sorted(missing)} measurement")
    build_commit = results[0].get("build_commit", "unknown")
    return by_kind, build_commit, peak_rss_kib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, help="quantinfer-cli binary")
    parser.add_argument("--llama-bench", required=True, help="llama-bench binary")
    parser.add_argument("--model", required=True, help="GGUF readable by both runtimes")
    parser.add_argument("--threads", default="1,2,4,8",
                        help="comma-separated thread counts")
    parser.add_argument("--generated-tokens", default=8, type=int)
    parser.add_argument("--repetitions", default=3, type=int,
                        help="llama-bench repetitions per thread count")
    parser.add_argument("--output", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    thread_counts = [int(value) for value in args.threads.split(",") if value]
    if not thread_counts or any(count < 1 for count in thread_counts):
        parser.error("--threads must list positive thread counts")

    runs = []
    llama_commit = "unknown"
    for threads in thread_counts:
        quantinfer = quantinfer_run(
            args.cli, args.model, threads, args.generated_tokens
        )
        llama, llama_commit, llama_rss = llama_run(
            args.llama_bench, args.model, threads,
            len(PROMPT_TOKENS.split(",")), args.generated_tokens, args.repetitions,
        )
        runs.append({
            "threads": threads,
            "quantinfer": quantinfer,
            "llama_cpp": {**llama, "peak_rss_kib": llama_rss},
            "decode_ratio_llama_over_quantinfer": (
                llama["decode_tokens_per_sec"]
                / quantinfer["profile_decode_tokens_per_sec"]
            ),
        })
        print(f"threads={threads} "
              f"quantinfer_decode_tps={quantinfer['profile_decode_tokens_per_sec']:.3f} "
              f"llama_decode_tps={llama['decode_tokens_per_sec']:.3f} "
              f"ratio={runs[-1]['decode_ratio_llama_over_quantinfer']:.2f}x")

    report = {
        "model": Path(args.model).name,
        "gguf_sha256": sha256(args.model),
        "cpu_model": cpu_model(),
        "cpu_arch": platform.machine(),
        "build_profile": "release",
        "prompt_tokens": len(PROMPT_TOKENS.split(",")),
        "generated_tokens": args.generated_tokens,
        "max_sequence_length": 64,
        "llama_cpp_build_commit": llama_commit,
        "peak_rss_note": (
            "QuantInfer peak RSS is VmHWM on Linux and ru_maxrss on macOS. "
            "llama.cpp peak RSS comes from /usr/bin/time."
        ),
        "runs": runs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
