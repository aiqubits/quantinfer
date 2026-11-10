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


def compare_inventory(cli, model, dtype, reference, architecture):
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
    assert actual["metadata"]["general.architecture"] == architecture
    assert actual_tensors["blk.0.attn_q.weight"][1] == dtype, "projection has unexpected dtype"
    if reference:
        source = GGUFReader(reference)
        for key in (
            "general.architecture", "llama.context_length", "llama.embedding_length",
            "llama.feed_forward_length", "llama.block_count", "llama.attention.head_count",
            "llama.attention.head_count_kv", "llama.rope.dimension_count", "llama.rope.freq_base",
            "tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.pre",
            "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.add_bos_token",
        ):
            assert actual["metadata"][key] == source.fields[key].contents(), key
        for key in ("llama.attention.key_length", "llama.attention.value_length"):
            assert actual["metadata"][key] == source.fields.get(
                key, source.fields["llama.rope.dimension_count"]
            ).contents(), key
        assert {name: row[0] for name, row in actual_tensors.items()} == {
            tensor.name: tensor.shape.tolist() for tensor in source.tensors
        }, "exported tensor names or dimensions differ from the external checkpoint"
    print(f"{len(expected_metadata)} metadata entries and {len(expected_tensors)} tensors agree")


def quantinfer_tokens(cli, model, prompt, count, profile=None):
    context = ["--inference-profile", profile] if profile else ["--max-sequence-length", "64"]
    output = subprocess.check_output([
        cli, "generate", "--input", model, "--prompt", prompt,
        "--max-new-tokens", str(count), "--temperature", "0", *context,
    ], text=True)
    tokens = {
        key: ast.literal_eval(value)
        for key, value in (line.split("=", 1) for line in output.splitlines() if "=" in line)
        if key in ("prompt_tokens", "generated_tokens")
    }
    return tokens["prompt_tokens"], tokens["generated_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", required=True, choices=["F32", "BF16", "Q8_0", "Q4_0"])
    parser.add_argument("--architecture", choices=["llama", "qwen3"], default="llama")
    parser.add_argument("--metadata-reference")
    parser.add_argument("--package", help="also generate from a model package or QIF artifact")
    parser.add_argument("--inference-profile")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--count", default=6, type=int)
    args = parser.parse_args()
    if args.inference_profile and not args.package:
        parser.error("--inference-profile requires --package")
    if args.metadata_reference and args.architecture != "llama":
        parser.error("--metadata-reference is available for Llama exports")
    compare_inventory(args.cli, args.model, args.dtype, args.metadata_reference, args.architecture)
    encoded, generated = llama_tokens(args.server, args.model, args.prompt, args.count)
    assert len(generated) == args.count, f"reference generated only {len(generated)} tokens"
    inputs = [(args.model, None)]
    if args.package:
        inputs.append((args.package, args.inference_profile))
    for model, profile in inputs:
        actual = quantinfer_tokens(args.cli, model, args.prompt, args.count, profile)
        assert actual == (encoded, generated), (model, actual, encoded, generated)
    print(f"prompt_tokens={encoded} generated_tokens={generated}")


if __name__ == "__main__":
    main()
