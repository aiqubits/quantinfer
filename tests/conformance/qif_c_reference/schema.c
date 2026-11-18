#include "qif.h"
#include <string.h>

static const char *dtypes[] = {"F32",  "F16",  "BF16", "Q4_0",
                               "Q4_K", "Q5_K", "Q6_K", "Q8_0"};
static const char *schemes[] = {"None", "Q8_0", "Q4_0", "Q4_K",
                                "Q5_K", "Q6_K", "Mixed"};
static const size_t blocks[] = {1, 1, 1, 32, 256, 256, 256, 32};
static const size_t widths[] = {4, 2, 2, 18, 144, 176, 210, 34};
static const unsigned scheme_ids[] = {0, 0, 0, 2, 3, 4, 5, 1};
static const char *packings[] = {
    "none",    "none",    "none",    "nibble_split_half",
    "k_quant", "k_quant", "k_quant", "scalar"};
static int equals(Json value, const char *expected) {
  return same_bytes(string_bytes(value),
                    (Bytes){(const unsigned char *)expected, strlen(expected)});
}
static size_t choice(Json value, const char *const *choices, size_t count) {
  for (size_t i = 0; i < count; i++)
    if (equals(value, choices[i]))
      return i;
  CHECK(0);
}
static uint64_t uint32(Json value) {
  uint64_t result = integer(value);
  CHECK(result <= UINT32_MAX);
  return result;
}
static void valid_string(Json value, int nonempty) {
  Bytes data = string_bytes(value);
  CHECK(data.size <= 16 * 1024 * 1024 && (!nonempty || data.size));
}
static size_t list_length(Json value) {
  CHECK(json_object_is_type(value, json_type_array));
  return json_object_array_length(value);
}
static Json item(Json value, size_t index) {
  CHECK(index < list_length(value));
  return json_object_array_get_idx(value, index);
}
static void strings(Json value) {
  for (size_t i = 0; i < list_length(value); i++)
    valid_string(item(value, i), 0);
}
static void digest(Json value) {
  Bytes data = string_bytes(value);
  CHECK(data.size == 64);
  for (size_t i = 0; i < data.size; i++)
    CHECK((data.data[i] >= '0' && data.data[i] <= '9') ||
          (data.data[i] >= 'a' && data.data[i] <= 'f') ||
          (data.data[i] >= 'A' && data.data[i] <= 'F'));
}
static uint64_t shape_elements(Json shape) {
  CHECK(list_length(shape) <= 32);
  uint64_t elements = 1;
  for (size_t i = 0; i < list_length(shape); i++) {
    uint64_t dimension = integer(item(shape, i));
    CHECK(dimension && elements <= UINT64_MAX / dimension);
    elements *= dimension;
  }
  return elements;
}
static size_t descriptor(Json value) {
  keys(value, "storage_type dtype quantization block_size group_size "
              "scale_type zero_point_type quantization_axis logical_shape "
              "physical_layout packing backend_requirements");
  size_t dtype = choice(field(value, "dtype"), dtypes, 8),
         block = blocks[dtype];
  CHECK(equals(field(value, "storage_type"),
               block == 1 ? dtypes[dtype] : "Quantized"));
  CHECK(equals(field(value, "quantization"), schemes[scheme_ids[dtype]]));
  CHECK(field(value, "group_size") == NULL &&
        equals(field(value, "zero_point_type"), "None"));
  CHECK(equals(field(value, "packing"), packings[dtype]));
  Json shape = field(value, "logical_shape");
  uint64_t elements = shape_elements(shape);
  CHECK(elements % block == 0);
  Json layout = field(value, "physical_layout");
  CHECK(equals(layout, "row_major") || equals(layout, "ggml_output_rows"));
  Json axis = field(value, "quantization_axis");
  if (block == 1)
    CHECK(field(value, "block_size") == NULL &&
          field(value, "scale_type") == NULL && equals(axis, "None"));
  else {
    CHECK(integer(field(value, "block_size")) == block &&
          equals(field(value, "scale_type"), "F16"));
    if (json_object_is_type(axis, json_type_object)) {
      keys(axis, "Axis");
      CHECK(integer(field(axis, "Axis")) == 0 &&
            equals(layout, "ggml_output_rows") && list_length(shape) == 2);
    } else
      CHECK(equals(axis, "Tensor"));
  }
  strings(field(value, "backend_requirements"));
  return dtype;
}
void validate_model(Json model) {
  keys(model, "schema_version model_id architecture source_revision "
              "source_format tokenizer_revision tensors quantization "
              "execution_variants kernel_requirements checksums compatibility");
  CHECK(uint32(field(model, "schema_version")) == 1);
  valid_string(field(model, "model_id"), 1);
  valid_string(field(model, "architecture"), 1);
  valid_string(field(model, "source_format"), 1);
  if (field(model, "source_revision"))
    valid_string(field(model, "source_revision"), 0);
  if (field(model, "tokenizer_revision"))
    valid_string(field(model, "tokenizer_revision"), 0);
  size_t scheme = choice(field(model, "quantization"), schemes, 7);
  Json tensors = field(model, "tensors");
  CHECK(list_length(tensors) <= 1048576);
  int has_quantized = 0;
  for (size_t i = 0; i < list_length(tensors); i++) {
    Json tensor = item(tensors, i);
    keys(tensor, "name descriptor checksum_sha256");
    valid_string(field(tensor, "name"), 1);
    digest(field(tensor, "checksum_sha256"));
    for (size_t j = 0; j < i; j++)
      CHECK(!same(field(tensor, "name"), field(item(tensors, j), "name")));
    size_t dtype = descriptor(field(tensor, "descriptor"));
    if (blocks[dtype] != 1) {
      has_quantized = 1;
      CHECK(scheme == 6 || scheme == scheme_ids[dtype]);
    }
  }
  CHECK(scheme == 0 || has_quantized);
  Json variants = field(model, "execution_variants");
  CHECK(list_length(variants) <= 1048576);
  for (size_t i = 0; i < list_length(variants); i++) {
    Json variant = item(variants, i);
    keys(variant, "name quantization dtype runtime_provider kernel_provider");
    valid_string(field(variant, "name"), 1);
    valid_string(field(variant, "dtype"), 0);
    valid_string(field(variant, "runtime_provider"), 1);
    valid_string(field(variant, "kernel_provider"), 1);
    CHECK(equals(field(variant, "quantization"), schemes[scheme]));
    for (size_t j = 0; j < i; j++)
      CHECK(!same(field(variant, "name"), field(item(variants, j), "name")));
  }
  Json kernel = field(model, "kernel_requirements");
  keys(kernel, "operations alignment head_dimension maximum_sequence_length");
  strings(field(kernel, "operations"));
  const char *optional[] = {"alignment", "head_dimension",
                            "maximum_sequence_length"};
  for (size_t i = 0; i < 3; i++)
    if (field(kernel, optional[i]))
      integer(field(kernel, optional[i]));
  Json hashes = field(model, "checksums");
  CHECK(json_object_is_type(hashes, json_type_object));
  Bytes previous = {NULL, 0};
  int seen = 0;
  json_object_object_foreach(hashes, key, checksum) {
    Bytes path = string_bytes(key_text(key));
    CHECK(path.size <= 16 * 1024 * 1024);
    if (seen) {
      size_t length = previous.size < path.size ? previous.size : path.size;
      int order = memcmp(previous.data, path.data, length);
      CHECK(order < 0 || (order == 0 && previous.size < path.size));
    }
    previous = path;
    seen = 1;
    relative_path(path);
    digest(checksum);
  }
  Json compatibility = field(model, "compatibility");
  keys(compatibility,
       "minimum_quantinfer_version gguf_version required_features");
  if (field(compatibility, "minimum_quantinfer_version"))
    valid_string(field(compatibility, "minimum_quantinfer_version"), 0);
  if (field(compatibility, "gguf_version"))
    uint32(field(compatibility, "gguf_version"));
  strings(field(compatibility, "required_features"));
}
void validate_descriptor(Json tensor, Json manifest) {
  Json value = field(manifest, "descriptor");
  size_t dtype = descriptor(value);
  CHECK(integer(field(tensor, "dtype")) == dtype + 1 &&
        integer(field(tensor, "physical_encoding")) == dtype + 1);
  CHECK(integer(field(tensor, "quantization_scheme")) == scheme_ids[dtype]);
  CHECK(integer(field(tensor, "layout")) ==
        (equals(field(value, "physical_layout"), "row_major") ? 1u : 2u));
  CHECK(same(field(tensor, "shape"), field(value, "logical_shape")));
  uint64_t elements = shape_elements(field(value, "logical_shape"));
  CHECK(elements / blocks[dtype] <= UINT64_MAX / widths[dtype]);
  CHECK(integer(field(tensor, "raw_length")) ==
        elements / blocks[dtype] * widths[dtype]);
  Bytes expected = string_bytes(field(manifest, "checksum_sha256")),
        actual = string_bytes(field(tensor, "checksum_sha256"));
  CHECK(expected.size == actual.size);
  for (size_t i = 0; i < expected.size; i++) {
    unsigned c = expected.data[i];
    if (c >= 'A' && c <= 'F')
      c += 32;
    CHECK(c == actual.data[i]);
  }
}
