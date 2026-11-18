"""Publish the fixed-model native K-quant matrix from SHA-256-verified sources.

The model-specific F32 reference test must run before the quality matrix. This
script prepares artifacts; quality and publication parity are checked by the
Rust k_quant_conformance integration test.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("smollm2", "qwen3"), required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--gguf-only", action="store_true",
                        help="let the Rust matrix publish and release QIF per variant")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    manifest_name = "smollm2-135m" if args.model == "smollm2" else "qwen3-0.6b"
    manifest = json.loads(
        (repository / "model-zoo" / "manifests" / f"{manifest_name}.json").read_text()
    )
    sources = {}
    for entry in [*manifest["artifacts"], manifest["weights"]]:
        source = args.source_dir / entry["path"]
        actual = sha256(source)
        if actual != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {source}")
        sources[entry["path"]] = actual
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = args.output_dir / f"{args.model}-f32.gguf"
    if not baseline.is_file():
        raise ValueError(f"prepare and independently validate the F32 baseline: {baseline}")
    converter = args.converter.resolve()
    artifacts = {baseline.name: sha256(baseline)}

    def convert(*options):
        subprocess.run([str(converter), *map(str, options)], check=True)

    for variant in ("q4_k", "q5_k", "q6_k"):
        stem = f"{args.model}-{variant}"
        gguf = args.output_dir / f"{stem}.gguf"
        convert(
            "--config", args.source_dir / "config.json",
            "--weights", args.source_dir / "model.safetensors",
            "--tokenizer", args.source_dir / "tokenizer.json",
            "--gguf-layout", "quantinfer", "--quantize", variant,
            "--output-format", "gguf", "--output", gguf,
        )
        artifacts[gguf.name] = sha256(gguf)
        if args.gguf_only:
            continue
        single = args.output_dir / f"{stem}.qif"
        package = args.output_dir / f"{stem}-package"
        convert("from-gguf", "--input", gguf, "--output", single,
                "--qif-profile", "single", "--qif-codec", "none")
        convert("from-gguf", "--input", gguf, "--output", package,
                "--qif-profile", "package", "--qif-codec", "zstd",
                "--max-shard-bytes", "16777216")
        for artifact in [single, *sorted(package.iterdir())]:
            artifacts[str(artifact.relative_to(args.output_dir))] = sha256(artifact)
    report = {
        "source_commit": os.environ.get("GITHUB_SHA"),
        "model_id": manifest["model_id"],
        "source_revision": manifest["source_revision"],
        "source_sha256": sources,
        "converter_sha256": sha256(converter),
        "layout": "native_row_major",
        "profiles": ["gguf"] if args.gguf_only else ["gguf", "single_none", "package_zstd"],
        "artifact_sha256": artifacts,
    }
    (args.output_dir / f"{args.model}-k-quant-artifacts.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
