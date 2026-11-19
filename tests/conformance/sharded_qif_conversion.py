#!/usr/bin/env python3
"""Verify bounded SafeTensors-to-QIF conversion against the GGUF transcode."""

import argparse
import hashlib
from pathlib import Path
import resource
import subprocess
import tempfile


ADDRESS_SPACE_BYTES = 384 * 1024 * 1024


def limit_address_space():
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def inventory(path):
    if path.is_file():
        return {path.name: digest(path)}
    return {
        file.relative_to(path).as_posix(): digest(file)
        for file in path.rglob("*")
        if file.is_file()
    }


def run(command, *, bounded=False):
    subprocess.run(
        command,
        check=True,
        preexec_fn=limit_address_space if bounded else None,
        timeout=300,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("converter", "cli", "config", "tokenizer", "index", "checksums", "gguf"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    assert args.gguf.stat().st_size > ADDRESS_SPACE_BYTES, "the fixture must exceed the memory ceiling"
    for profile in ("single", "package"):
        for codec in ("none", "zstd", "lz4"):
            options = ["--qif-profile", profile, "--qif-codec", codec]
            if profile == "package":
                options.extend(["--max-shard-bytes", "67108864"])
            with tempfile.TemporaryDirectory(prefix="qif-sharded-conversion-") as directory:
                root = Path(directory)
                (root / "sharded").mkdir()
                (root / "baseline").mkdir()
                output = root / "sharded/model.qif"
                reference = root / "baseline/model.qif"
                run([
                    args.converter, "--config", args.config, "--tokenizer", args.tokenizer,
                    "--weights", args.index, "--weights-checksums", args.checksums,
                    "--output", output, *options,
                ], bounded=True)
                run([
                    args.converter, "from-gguf", "--input", args.gguf,
                    "--model-id", "model", "--output", reference, *options,
                ])
                produced = inventory(output)
                assert produced and produced == inventory(reference), (profile, codec)
                run([args.cli, "verify", "--input", output, "--json"], bounded=True)
                print(f"{profile}/{codec}: {len(produced)} files match under a 384 MiB address-space limit", flush=True)


if __name__ == "__main__":
    main()
