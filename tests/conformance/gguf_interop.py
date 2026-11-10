"""Compare the QuantInfer CLI with a pinned llama.cpp build on the same GGUF."""

import argparse
import ast
import json
import math
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from gguf import GGUFReader


def request(url, data=None, timeout=30):
    headers = {"Content-Type": "application/json"} if data is not None else {}
    payload = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(
        urllib.request.Request(url, data=payload, headers=headers), timeout=timeout
    ) as response:
        return json.load(response)


def llama_tokens(server, model, prompt, count):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    endpoint = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [
                server, "--model", model, "--host", "127.0.0.1", "--port", port,
                "--ctx-size", "64", "--parallel", "1", "--threads", "1",
                "--n-gpu-layers", "0", "--cache-type-k", "f32", "--cache-type-v", "f32",
            ],
            stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while process.poll() is None and time.monotonic() < deadline:
                try:
                    if request(f"{endpoint}/health", timeout=2)["status"] == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.2)
            else:
                log.seek(0)
                raise RuntimeError(f"llama.cpp could not load {model}\n{log.read()}")
            encoded = request(f"{endpoint}/tokenize", {
                "content": prompt, "add_special": False, "parse_special": True,
            })["tokens"]
            completion = request(f"{endpoint}/completion", {
                "prompt": encoded, "temperature": 0.0, "n_predict": count,
                "return_tokens": True, "cache_prompt": False,
            })
            return encoded, completion["tokens"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def compare_inventory(cli, model, dtype):
    actual = json.loads(subprocess.check_output(
        [cli, "inspect-gguf", "--input", model, "--json"], text=True,
    ))
    upstream = GGUFReader(model)
    expected_metadata = {
        key: field.contents() for key, field in upstream.fields.items()
        if not key.startswith("GGUF.")
    }
    assert actual["metadata"].keys() == expected_metadata.keys(), "metadata keys differ"
    for key, expected in expected_metadata.items():
        value = actual["metadata"][key]
        if isinstance(expected, float):
            assert math.isclose(value, expected, rel_tol=1e-6), (key, value, expected)
        else:
            assert value == expected, f"metadata differs at {key}"
    expected_tensors = {
        tensor.name: (tensor.shape.tolist(), tensor.tensor_type.name, tensor.n_bytes)
        for tensor in upstream.tensors
    }
    actual_tensors = {
        tensor["name"]: (tensor["shape"], tensor["dtype"], tensor["size"])
        for tensor in actual["tensors"]
    }
    assert actual_tensors == expected_tensors, "tensor shapes, types or sizes differ"
    assert actual["metadata"]["general.architecture"] == "llama"
    assert actual_tensors["blk.0.attn_q.weight"][1] == dtype, "projection has unexpected dtype"
    print(f"{len(expected_metadata)} metadata entries and {len(expected_tensors)} tensors agree")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", required=True, choices=["F32", "Q8_0", "Q4_0"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--count", default=6, type=int)
    args = parser.parse_args()
    compare_inventory(args.cli, args.model, args.dtype)
    output = subprocess.check_output([
        args.cli, "generate", "--input", args.model, "--prompt", args.prompt,
        "--max-new-tokens", str(args.count), "--max-sequence-length", "64",
        "--temperature", "0",
    ], text=True)
    actual = {
        key: ast.literal_eval(value)
        for key, value in (line.split("=", 1) for line in output.splitlines() if "=" in line)
        if key in ("prompt_tokens", "generated_tokens")
    }
    encoded, generated = llama_tokens(args.server, args.model, args.prompt, args.count)
    assert actual["prompt_tokens"] == encoded, (actual["prompt_tokens"], encoded)
    assert len(generated) == args.count, f"reference generated only {len(generated)} tokens"
    assert actual["generated_tokens"] == generated, (actual["generated_tokens"], generated)
    print(f"prompt_tokens={encoded} generated_tokens={generated}")


if __name__ == "__main__":
    main()
