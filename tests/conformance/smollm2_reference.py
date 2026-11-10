"""Emit the SmolLM2-135M golden fixture with Transformers and CPU PyTorch."""

import argparse
import json

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Local model at the revision in the model-zoo manifest")
    args = parser.parse_args()

    torch.set_num_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float32, attn_implementation="eager", local_files_only=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)

    @torch.inference_mode()
    def last_logits(token_ids):
        inputs = torch.tensor([token_ids])
        return model(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            position_ids=torch.arange(len(token_ids)).unsqueeze(0),
            use_cache=False,
        ).logits[0, -1]

    prompt = "The capital of France is"
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prefill = last_logits(prompt_tokens)
    top_10 = prefill.topk(10).indices.tolist()
    sample_ids = sorted(set(top_10 + prompt_tokens + [
        0, 1, 2, 3, 1024, 4096, 8192, 16384, 32768, 49151,
    ]))
    generated = []
    for _ in range(8):
        next_token = last_logits(prompt_tokens + generated).argmax().item()
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
        "eos_token_id": tokenizer.eos_token_id,
        "top_10": top_10,
        "logits": [[index, round(prefill[index].item(), 8)] for index in sample_ids],
        "generated_tokens": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
