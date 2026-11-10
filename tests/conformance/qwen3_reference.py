"""Generate F32 Qwen3-0.6B reference logits from the pinned SafeTensors."""

import argparse
import json

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Local model directory verified against its manifest")
    args = parser.parse_args()

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()

    prompt = "What is 2 + 2? Reply with only the number."
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )
    generated = []
    with torch.inference_mode():
        for _ in range(16):
            inputs = torch.tensor([prompt_tokens + generated])
            logits = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                position_ids=torch.arange(inputs.size(1)).unsqueeze(0),
                use_cache=False,
            ).logits[0, -1]
            if not generated:
                prefill = logits.tolist()
            next_token = logits.argmax().item()
            generated.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    print(json.dumps({
        "reference": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": "float32",
            "attention": "eager",
            "threads": 1,
        },
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "logits": prefill,
        "eos_token_id": tokenizer.eos_token_id,
        "generated_tokens": generated,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
