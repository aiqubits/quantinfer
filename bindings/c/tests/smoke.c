#include "quantinfer.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static void require_ok(quantinfer_status status) {
    if (status != QUANTINFER_OK) {
        fprintf(stderr, "%s\n", quantinfer_last_error());
        assert(status == QUANTINFER_OK);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2);
    assert(quantinfer_abi_version() == 1);
    quantinfer_model *model = NULL;
    assert(quantinfer_model_open(NULL, &model) == QUANTINFER_INVALID_ARGUMENT);
    assert(model == NULL && strlen(quantinfer_last_error()) != 0);
    require_ok(quantinfer_model_open(argv[1], &model));

    const char *prompt = "The capital of France is";
    const uint32_t expected_prompt[] = {504, 3575, 282, 4649, 314};
    uint32_t tokens[5];
    size_t token_count = 0;
    assert(quantinfer_tokenize(model, prompt, NULL, 0, &token_count) == QUANTINFER_BUFFER_TOO_SMALL);
    assert(token_count == 5);
    require_ok(quantinfer_tokenize(model, prompt, tokens, 5, &token_count));
    assert(memcmp(tokens, expected_prompt, sizeof(tokens)) == 0);

    quantinfer_generation_options options = quantinfer_generation_defaults();
    options.max_new_tokens = 8;
    const uint32_t expected_output[] = {260, 3575, 282, 260, 1798, 30, 198, 198};
    uint32_t output[8];
    size_t count = 0;
    assert(quantinfer_generate_tokens(model, tokens, token_count, &options, NULL, 0, &count)
           == QUANTINFER_BUFFER_TOO_SMALL);
    assert(count == 8);
    require_ok(quantinfer_generate_tokens(model, tokens, token_count, &options, output, 8, &count));
    assert(count == 8 && memcmp(output, expected_output, sizeof(output)) == 0);

    const char *expected_text = " the capital of the country.\n\n";
    char text[64];
    size_t text_size = 0;
    assert(quantinfer_detokenize(model, output, count, NULL, 0, &text_size)
           == QUANTINFER_BUFFER_TOO_SMALL);
    assert(text_size == strlen(expected_text) + 1);
    require_ok(quantinfer_detokenize(model, output, count, text, sizeof(text), &text_size));
    assert(strcmp(text, expected_text) == 0);

    options.eos_token_id = expected_output[0];
    require_ok(quantinfer_generate_tokens(model, tokens, token_count, &options, output, 8, &count));
    assert(count == 1 && output[0] == options.eos_token_id);
    quantinfer_model_free(model);
    return 0;
}
