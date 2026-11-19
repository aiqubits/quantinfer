"""Exercise Studio jobs and model management with the pinned SmolLM2 fixture."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from http_smoke import server


def call(base, route, *, method="GET", body=None, expect=200, origin=True, host=None):
    headers = {"X-QuantInfer-Studio": "1"} if origin else {}
    if host:
        headers["Host"] = host
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    request = urllib.request.Request(base + route, data=data, headers=headers, method=method)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        content = response.read()
        assert response.code == expect, (route, response.code, content[:500])
        return json.loads(content) if content else None


def submit(base, **request):
    return call(base, "/studio/api/jobs", method="POST", body=request, expect=202)["id"]


def wait(base, identifier, phase, timeout=360):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(job for job in call(base, "/studio/api/jobs")["jobs"] if job["id"] == identifier)
        if job["phase"] == phase:
            return job
        assert job["phase"] in {"queued", "running"}, job
        time.sleep(0.15)
    raise AssertionError((identifier, phase, job))


def check(args):
    with tempfile.TemporaryDirectory(prefix="quantinfer-studio-smoke-") as directory:
        workspace = Path(directory) / "studio"
        server_args = ["--studio-dir", workspace, "--studio-converter", args.converter, "--studio-cli", args.cli]
        with server(args.server, *server_args) as (base, _, _):
            shutil.copyfile(args.gguf, workspace / "sources/smol.gguf")
            shutil.copytree(args.lora, workspace / "adapters/peft")
            call(base, "/studio/api/jobs", host="unrelated.invalid", expect=403)
            prompts = {"text": '"Write a short answer."\n'}
            dataset = "/studio/api/datasets/calibration.jsonl"
            call(base, dataset, method="PUT", body=prompts, origin=False, expect=403)
            call(base, dataset, method="PUT", body={"text": '{"invalid":1}'}, expect=400)
            call(base, dataset, method="PUT", body=prompts, expect=204)
            assert call(base, "/studio/api/inventory")["datasets"] == ["calibration.jsonl"]
            inputs = {"source": "smol.gguf", "dataset": "calibration.jsonl", "context": 32}
            calibration = wait(base, submit(base, kind="calibrate", **inputs), "succeeded")
            assert calibration["result"]["tensor_count"] == 210
            assert (workspace / "artifacts" / calibration["id"] / "imatrix.json").is_file()

            (workspace / "datasets/bad.jsonl").write_text("invalid\n")
            failed = submit(base, kind="calibrate", **(inputs | {"dataset": "bad.jsonl"}))
            assert "invalid calibration JSON" in wait(base, failed, "failed")["detail"]
            assert not (workspace / "artifacts" / failed).exists()
            quantize = {"kind": "quantize", **inputs, "bits": 4, "group_size": 32, "codec": "zstd", "package": True}
            cancelled = submit(base, algorithm="gptq", **quantize)
            wait(base, cancelled, "running")
            call(base, f"/studio/api/jobs/{cancelled}", method="DELETE", expect=202)
            wait(base, cancelled, "cancelled")
            assert not (workspace / "artifacts" / cancelled).exists()

            quantized = wait(base, submit(base, algorithm="awq", **quantize), "succeeded")
            assert quantized["result"]["artifact"] == "model.qif"
            original = workspace / "artifacts" / quantized["id"] / "model.qif"
            repacked = wait(base, submit(base, kind="package", source=f"job:{quantized['id']}",
                                         codec="lz4", max_shard_bytes=16 * 1024 * 1024), "succeeded")
            assert repacked["result"]["tensor_count"] == 272
            source = f"job:{repacked['id']}"
            package = workspace / "artifacts" / repacked["id"] / "model.qif"
            verification = wait(base, submit(base, kind="verify", source=source), "succeeded")
            assert verification["result"]["verified"] is True
            before = json.loads((original / "qif.json").read_bytes())
            after = json.loads((package / "qif.json").read_bytes())
            assert before["root_hash"] != after["root_hash"]
            assert before["model"] == after["model"]
            assert (package / "metadata.qmb").read_bytes() == (original / "metadata.qmb").read_bytes()

            profile_path = "/studio/api/profiles/short-context"
            configuration = {"source": source, "context_length": 12, "thread_count": 1}
            profile = call(base, profile_path, method="PUT", body=configuration)
            assert profile["settings"]["quantization_variant"] == "q4_0"
            call(base, profile_path, method="PUT", body=configuration | {"memory_budget_bytes": 1}, expect=400)
            assert call(base, profile_path) == profile
            registration = {"source": source, "profile": "short-context", "lora": "peft"}
            call(base, "/studio/api/models/smol", method="PUT", body=registration, expect=204)
            call(base, "/studio/api/models/smol", method="PUT", body=registration | {"source": "smol.gguf"}, expect=400)
            info = call(base, "/studio/api/registry")["models"][0]
            assert info["managed"] and info["has_lora"] and info["applied_profile"] == profile["settings"]
            generation = {"model": "smol", "tokens": [1, 2], "max_new_tokens": 2}
            with_adapter = call(base, "/v1/generate", method="POST", body=generation)["tokens"]
            expected = subprocess.check_output([
                args.cli, "generate", "--input", original, "--lora", args.lora, "--tokens", "1,2",
                "--max-sequence-length", "12", "--max-new-tokens", "2",
            ], text=True)
            assert f"generated_tokens={with_adapter}" in expected
            call(base, "/v1/generate", method="POST", body=generation | {"tokens": [1] * 13}, expect=400)

            benchmark = wait(base, submit(base, kind="benchmark", source=source,
                                           prompt="--profile-report", tokens=4, context=32), "succeeded")
            result = benchmark["result"]
            assert result["metrics"]["decode_tokens"] == 4
            assert result["artifact_sha256"] == profile["artifact_sha256"]
            assert len(result["cli_sha256"]) == 64 and result["artifact_digest_scope"] == "manifest"
            assert not any(path.name.startswith(".") for path in (workspace / "artifacts").iterdir())

        with server(args.server, *server_args) as (base, _, _):
            assert call(base, "/studio/api/profiles/short-context") == profile
            assert call(base, "/studio/api/jobs")["jobs"][0]["id"] == benchmark["id"]
            assert call(base, "/v1/generate", method="POST", body=generation)["tokens"] == with_adapter
            call(base, "/studio/api/models/smol", method="DELETE", expect=204)
            call(base, "/v1/generate", method="POST", body=generation, expect=404)
            call(base, profile_path, method="DELETE", expect=204)
            assert package.is_dir()
        print("Studio calibration, cancellation, AWQ publication, repacking, profile and LoRA serving, benchmark and restart checks passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("server", "cli", "converter", "gguf", "lora"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    check(parser.parse_args())


if __name__ == "__main__":
    main()
