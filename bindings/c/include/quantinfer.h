#ifndef QUANTINFER_H
#define QUANTINFER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct quantinfer_model quantinfer_model;
typedef int32_t quantinfer_status;

enum {
    QUANTINFER_OK = 0,
    QUANTINFER_INVALID_ARGUMENT = 1,
    QUANTINFER_BUFFER_TOO_SMALL = 2,
    QUANTINFER_RUNTIME_ERROR = 3
};

#define QUANTINFER_EOS_AUTO UINT32_MAX
#define QUANTINFER_EOS_DISABLED (UINT32_MAX - 1U)

typedef struct {
    size_t context_length; /* Zero chooses prompt length plus max_new_tokens. */
    size_t max_new_tokens;
    float temperature;
    size_t top_k; /* Zero disables top-k sampling. */
    float top_p; /* Zero disables top-p sampling. */
    uint32_t eos_token_id;
    uint64_t seed;
} quantinfer_generation_options;

uint32_t quantinfer_abi_version(void);
quantinfer_generation_options quantinfer_generation_defaults(void);

/* The last failure on this thread, valid until its next failure or exit. */
const char *quantinfer_last_error(void);

/* Paths and input text must be NUL-terminated UTF-8. */
quantinfer_status quantinfer_model_open(const char *path, quantinfer_model **out_model);

/* Free once after all operations on the handle return. NULL is allowed. */
void quantinfer_model_free(quantinfer_model *model);

/* The count reports the required size on QUANTINFER_BUFFER_TOO_SMALL. Pass
 * NULL and zero capacity to query the size. Detokenize includes the final NUL.
 * Buffers must be aligned, disjoint from other arguments and valid for the
 * stated capacity. Model handles may be shared by concurrent operations. */
quantinfer_status quantinfer_tokenize(const quantinfer_model *model, const char *text,
                                     uint32_t *output, size_t capacity, size_t *written);
quantinfer_status quantinfer_detokenize(const quantinfer_model *model, const uint32_t *tokens,
                                       size_t count, char *output, size_t capacity,
                                       size_t *written);

/* Generation requires capacity for max_new_tokens. NULL options use defaults. */
quantinfer_status quantinfer_generate_tokens(const quantinfer_model *model,
                                            const uint32_t *tokens, size_t count,
                                            const quantinfer_generation_options *options,
                                            uint32_t *output, size_t capacity, size_t *written);

#ifdef __cplusplus
}
#endif

#endif
