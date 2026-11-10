"""Write deterministic PEFT factors for a Llama or Qwen3 decoder."""

import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = json.loads(args.config.read_text())
    hidden = model["hidden_size"]
    vocabulary = model["vocab_size"]
    heads = model["num_attention_heads"]
    query = heads * (model.get("head_dim") or hidden // heads)

    args.output.mkdir()
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "bias": "none",
        "target_modules": r"(?:lm_head|model\.layers\.0\.self_attn\.q_proj)",
        "r": 1,
        "lora_alpha": 4,
        "rank_pattern": {"q_proj": 2},
        "alpha_pattern": {"q_proj": 1},
        "use_rslora": True,
    }
    (args.output / "adapter_config.json").write_text(json.dumps(config))

    header = {"__metadata__": {"format": "pt"}}
    data = bytearray()

    def add(name, rows, columns, values):
        assert len(values) == rows * columns
        start = len(data)
        data.extend(struct.pack(f"<{len(values)}f", *values))
        header[f"base_model.model.{name}.weight"] = {
            "dtype": "F32",
            "shape": [rows, columns],
            "data_offsets": [start, len(data)],
        }

    add("lm_head.lora_A", 1, hidden, [1.0] + [0.0] * (hidden - 1))
    logits = [0.0] * vocabulary
    logits[42], logits[43] = 1000.0, -1000.0
    add("lm_head.lora_B", vocabulary, 1, logits)
    projection = "model.layers.0.self_attn.q_proj"
    add(f"{projection}.lora_A", 2, hidden, [0.001, -0.002] * hidden)
    add(f"{projection}.lora_B", query, 2, [0.001, -0.002] * query)

    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with (args.output / "adapter_model.safetensors").open("wb") as file:
        file.write(struct.pack("<Q", len(encoded)))
        file.write(encoded)
        file.write(data)


if __name__ == "__main__":
    main()
