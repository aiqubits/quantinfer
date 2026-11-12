"""Exercise the HTTP service with a pinned SmolLM2 model."""

import concurrent.futures
import contextlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


@contextlib.contextmanager
def server(binary, *args):
    with tempfile.TemporaryDirectory() as logs:
        log_path = Path(logs) / "server.log"
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [binary, "--bind", "127.0.0.1:0", "--context-length", "64", *args],
                stdout=log, stderr=log,
            )
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                addresses = [line.split()[-1] for line in log_path.read_text().splitlines()
                             if "listening on" in line]
                if addresses:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(f"server did not start\n{log_path.read_text()}")
            yield f"http://{addresses[0]}", process, log_path
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
                raise RuntimeError(f"server exited with {exit_code}\n{log_path.read_text()}")


def request(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if data is None else {"Content-Type": "application/json"}
    with urllib.request.urlopen(
        urllib.request.Request(base + path, data=data, headers=headers), timeout=30
    ) as response:
        return response.read().decode()


def check(binary, model, golden, adapter=None):
    prompt = golden["prompt"]
    expected = golden["generated_tokens"]
    with server(binary, "--model", f"primary={model}", "--model", f"secondary={model}") as (base, _, _):
        assert json.loads(request(base, "/healthz")) == {"status": "ok"}
        models = json.loads(request(base, "/v1/models"))["data"]
        assert [entry["id"] for entry in models] == ["primary", "secondary"]
        studio = request(base, "/studio")
        assert "QuantInfer Studio" in studio
        assert "/v1/generate" in studio
        assert "Live generation samples" in studio
        registry = json.loads(request(base, "/studio/api/registry"))["models"]
        assert [entry["id"] for entry in registry] == ["primary", "secondary"]
        assert all(entry["package"] is False for entry in registry)
        common = {"prompt": prompt, "max_new_tokens": len(expected)}
        for stream_model in ("secondary", "primary"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                ordinary = executor.submit(
                    request, base, "/v1/generate", {**common, "model": "primary"}
                )
                streamed = executor.submit(
                    request, base, "/v1/generate", {**common, "model": stream_model, "stream": True}
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
        counters = {name: int(value) for name, value in (line.split() for line in metrics.splitlines())}
        assert counters["quantinfer_requests_total"] == 4
        assert counters["quantinfer_generated_tokens_total"] == 4 * len(expected)
        assert counters["quantinfer_batched_forward_steps_total"] > 0

    with server(binary, "--model", f"primary={model}", "--kv-budget-bytes", "1") as (base, _, _):
        try:
            request(base, "/v1/generate", {"model": "primary", "prompt": prompt})
        except urllib.error.HTTPError as error:
            assert error.code == 429
            assert "KV quota" in json.load(error)["error"]
        else:
            raise AssertionError("server admitted a request over the KV quota")

    with server(binary, "--model", f"primary={model}", "--request-timeout-ms", "1") as (base, _, _):
        try:
            request(base, "/v1/generate", {"model": "primary", "prompt": prompt})
        except urllib.error.HTTPError as error:
            assert error.code == 504
        else:
            raise AssertionError("server completed a request beyond its deadline")

    with server(binary, "--requests-per-second", "1") as (base, _, _):
        for expected_status in (404, 429):
            try:
                request(base, "/v1/generate", {"model": "unknown", "tokens": [1]})
            except urllib.error.HTTPError as error:
                assert error.code == expected_status
                if error.code == 429:
                    assert error.headers["Retry-After"] == "1"
            else:
                raise AssertionError("unexpected successful response")

    if hasattr(signal, "SIGHUP"):
        check_registry(binary, model, golden, adapter)


def check_registry(binary, model, golden, adapter):
    with tempfile.TemporaryDirectory() as directory:
        config = Path(directory) / "registry.json"
        (config.parent / "model.gguf").symlink_to(Path(model).resolve())
        config.write_text(json.dumps({"primary": {"path": "model.gguf"}}))
        with server(binary, "--model-registry", str(config)) as (base, process, log):
            def ids():
                return [entry["id"] for entry in json.loads(request(base, "/v1/models"))["data"]]

            def reload(entries, message):
                log_start = len(log.read_text())
                config.write_text(json.dumps(entries))
                process.send_signal(signal.SIGHUP)
                deadline = time.monotonic() + 30
                while message not in log.read_text()[log_start:] and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert message in log.read_text()[log_start:], log.read_text()

            assert ids() == ["primary"]
            reload({"secondary": {"path": "missing.gguf"}}, "model registry reload failed")
            assert ids() == ["primary"]
            reload({"secondary": {"path": "model.gguf", "revision": "2"}}, "model registry reloaded")
            assert ids() == ["secondary"]
            result = json.loads(request(base, "/v1/generate", {
                "model": "secondary", "prompt": golden["prompt"], "max_new_tokens": 2,
            }))
            assert result["tokens"] == golden["generated_tokens"][:2]
            if adapter:
                (config.parent / "adapter").symlink_to(Path(adapter).resolve(), target_is_directory=True)
                entry = {"path": "model.gguf", "lora": "adapter"}
                base_entry = {"path": "model.gguf", "revision": "2"}
                reload({"adapted": entry, "secondary": base_entry}, "model registry reloaded")
                assert ids() == ["adapted", "secondary"]
                payload = {"model": "adapted", "prompt": golden["prompt"], "max_new_tokens": 1}
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    adapted = executor.submit(request, base, "/v1/generate", payload)
                    original = executor.submit(request, base, "/v1/generate", {**payload, "model": "secondary"})
                    assert json.loads(adapted.result())["tokens"] == [43]
                    assert json.loads(original.result())["tokens"] == golden["generated_tokens"][:1]
                reload({"adapted": {**entry, "lora": "missing-adapter"}}, "model registry reload failed")
                result = json.loads(request(base, "/v1/generate", payload))
                assert result["tokens"] == [43]


if __name__ == "__main__":
    check(sys.argv[1], sys.argv[2], json.loads(Path(sys.argv[3]).read_text()), *sys.argv[4:])
