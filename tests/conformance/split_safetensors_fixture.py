"""Split a verified SafeTensors fixture into two shards for conformance tests."""

import json
from pathlib import Path
import struct
import sys


def split(source_path, destination):
    destination.mkdir(exist_ok=True)
    weight_map = {}
    with source_path.open("rb") as source:
        (header_size,) = struct.unpack("<Q", source.read(8))
        header = json.loads(source.read(header_size))
        data_start = source.tell()
        header.pop("__metadata__", None)
        tensors = list(header.items())
        for shard, group in enumerate((tensors[::2], tensors[1::2]), 1):
            name = f"model-{shard:05d}-of-00002.safetensors"
            shard_header = {}
            shard_size = 0
            for tensor, info in group:
                start, end = info["data_offsets"]
                shard_header[tensor] = {
                    **info,
                    "data_offsets": [shard_size, shard_size + end - start],
                }
                shard_size += end - start
                weight_map[tensor] = name

            encoded = json.dumps(shard_header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            with (destination / name).open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                for tensor, info in group:
                    start, end = info["data_offsets"]
                    source.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError(f"truncated tensor {tensor}")
                        output.write(chunk)
                        remaining -= len(chunk)

    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )


if __name__ == "__main__":
    split(Path(sys.argv[1]), Path(sys.argv[2]))
