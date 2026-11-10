"""Exercise the HTTP service with a pinned SmolLM2 model."""

import concurrent.futures
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


@contextlib.contextmanager
def server(binary, *args):
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(
            [binary, "--bind", "127.0.0.1:0", "--context-length", "64", *args],
            stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                log.seek(0)
                addresses = [line.split()[-1] for line in log if "listening on" in line]
                if addresses:
                    break
                time.sleep(0.05)
            else:
                log.seek(0)
                raise RuntimeError(f"server did not start\n{log.read()}")
            yield f"http://{addresses[0]}"
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            if exit_code != 0:
                log.seek(0)
                raise RuntimeError(f"server exited with {exit_code}\n{log.read()}")


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urllib.request.urlopen(
        urllib.request.Request(base + path, data=data, headers=headers), timeout=30
    ) as response:
        return response.read().decode()


def check(binary, model, golden):
    prompt = golden["prompt"]
    expected = golden["generated_tokens"]
    with server(binary, "--model", f"primary={model}", "--model", f"secondary={model}") as base:
        assert json.loads(request(base, "/healthz")) == {"status": "ok"}
        models = json.loads(request(base, "/v1/models"))["data"]
        assert [entry["id"] for entry in models] == ["primary", "secondary"]
        common = {"prompt": prompt, "max_new_tokens": len(expected)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ordinary = executor.submit(
                request, base, "/v1/generate", {**common, "model": "primary"}
            )
            streamed = executor.submit(
                request, base, "/v1/generate", {**common, "model": "secondary", "stream": True}
            )
            result = json.loads(ordinary.result())
            events = [json.loads(line[6:]) for line in streamed.result().splitlines()
                      if line.startswith("data: ")]
        assert result["tokens"] == expected
        assert result["text"] == golden["generated_text"]
        assert result["finish_reason"] == "length"
        assert [event["token_id"] for event in events] == expected
        assert events[-1]["finish_reason"] == "length"
        assert events[-1]["request_id"] != result["request_id"]
        metrics = request(base, "/metrics")
        assert "quantinfer_requests_total 2\n" in metrics
        assert f"quantinfer_generated_tokens_total {2 * len(expected)}\n" in metrics

    with server(binary, "--model", f"primary={model}", "--kv-budget-bytes", "1") as base:
        try:
            request(base, "/v1/generate", {"model": "primary", "prompt": prompt})
        except urllib.error.HTTPError as error:
            assert error.code == 429
            assert "KV quota" in json.load(error)["error"]
        else:
            raise AssertionError("server admitted a request over the KV quota")

    with server(binary, "--model", f"primary={model}", "--request-timeout-ms", "1") as base:
        try:
            request(base, "/v1/generate", {"model": "primary", "prompt": prompt})
        except urllib.error.HTTPError as error:
            assert error.code == 504
        else:
            raise AssertionError("server completed a request beyond its deadline")


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()))
