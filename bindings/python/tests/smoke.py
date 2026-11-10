import json
from pathlib import Path
import sys

from quantinfer import Model, QuantInferError


def main(model_path, golden_path):
    golden = json.loads(Path(golden_path).read_text(encoding="utf-8"))
    prompt = golden["prompt"]
    expected = golden["generated_tokens"]
    with Model(Path(model_path)) as model:
        tokens = model.tokenize(prompt)
        assert tokens == golden["prompt_tokens"]
        assert model.tokenize("") == []
        assert model.detokenize([]) == ""
        assert model.generate_tokens(tokens, max_new_tokens=len(expected)) == expected
        assert model.detokenize(expected) == golden["generated_text"]
        assert model.generate(prompt, max_new_tokens=len(expected)) == golden["generated_text"]
        assert model.generate_tokens(tokens, eos_token_id=expected[0]) == expected[:1]
        assert model.generate_tokens(
            tokens, max_new_tokens=len(expected), eos_token_id=None
        ) == expected
        try:
            model.generate_tokens(tokens, max_new_tokens=-1)
        except ValueError:
            pass
        else:
            raise AssertionError("negative max_new_tokens was accepted")
    model.close()
    try:
        model.tokenize(prompt)
    except QuantInferError as error:
        assert str(error) == "model is closed"
    else:
        raise AssertionError("a closed model accepted a request")


if __name__ == "__main__":
    main(*sys.argv[1:])
