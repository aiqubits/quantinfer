#include "qif.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct Resource {
  void *value;
  void (*release)(void *);
  struct Resource *next;
} Resource;
static Resource *resources;
jmp_buf qif_error;
_Noreturn void fail(const char *file, int line) {
  fprintf(stderr, "qif-c-reference: invalid/unsupported input (%s:%d)\n", file,
          line);
  longjmp(qif_error, 1);
}
void *owned(void *value, void (*release)(void *)) {
  CHECK(value);
  Resource *entry = malloc(sizeof(*entry));
  if (!entry) {
    release(value);
    CHECK(0);
  }
  *entry = (Resource){value, release, resources};
  resources = entry;
  return value;
}
void *allocate(size_t size) {
  CHECK(size <= LIMIT * 4u);
  return owned(calloc(size ? size : 1, 1), free);
}
void cleanup(void) {
  while (resources) {
    Resource *entry = resources;
    resources = entry->next;
    entry->release(entry->value);
    free(entry);
  }
}
static void release_json(void *value) { json_object_put(value); }
Json own_json(Json value) { return owned(value, release_json); }
Json object(void) { return own_json(json_object_new_object()); }
Json array(void) { return own_json(json_object_new_array()); }
Json text(Bytes value) {
  CHECK(value.size <= LIMIT * 2u);
  return own_json(
      json_object_new_string_len((const char *)value.data, (int)value.size));
}
Json string(const char *value) {
  return text((Bytes){(const unsigned char *)value, strlen(value)});
}
Json number(uint64_t value) { return own_json(json_object_new_uint64(value)); }
void put(Json parent, const char *key, Json value) {
  CHECK(json_object_object_add(parent, key, json_object_get(value)) == 0);
}
void push(Json parent, Json value) {
  CHECK(json_object_array_add(parent, json_object_get(value)) == 0);
}
Json field(Json parent, const char *key) {
  Json value = NULL;
  CHECK(json_object_is_type(parent, json_type_object));
  CHECK(json_object_object_get_ex(parent, key, &value));
  return value;
}
uint64_t integer(Json value) {
  CHECK(json_object_is_type(value, json_type_int));
  CHECK(json_object_get_int64(value) >= 0);
  return json_object_get_uint64(value);
}
Bytes string_bytes(Json value) {
  CHECK(json_object_is_type(value, json_type_string));
  return (Bytes){(const unsigned char *)json_object_get_string(value),
                 (size_t)json_object_get_string_len(value)};
}
int same_bytes(Bytes left, Bytes right) {
  return left.size == right.size &&
         (left.size == 0 || memcmp(left.data, right.data, left.size) == 0);
}
int same(Json left, Json right) {
  if (json_object_is_type(left, json_type_string) &&
      json_object_is_type(right, json_type_string))
    return same_bytes(string_bytes(left), string_bytes(right));
  return json_object_equal(left, right);
}
void keys(Json value, const char *expected) {
  CHECK(json_object_is_type(value, json_type_object));
  const char *cursor = expected;
  json_object_object_foreach(value, key, entry) {
    (void)entry;
    CHECK(*cursor);
    size_t length = strcspn(cursor, " ");
    CHECK(strlen(key) == length && strncmp(key, cursor, length) == 0);
    cursor += length;
    if (*cursor == ' ')
      cursor++;
  }
  CHECK(*cursor == 0);
}
Bytes serialized(Json value) {
  const char *data = json_object_to_json_string_ext(
      value, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
  CHECK(data);
  return json_keys((Bytes){(const unsigned char *)data, strlen(data)}, 0);
}
void utf8(Bytes value) {
  for (size_t i = 0; i < value.size; i++) {
    unsigned c = value.data[i];
    if (c < 128)
      continue;
    size_t continuation = c >= 0xc2 && c <= 0xdf   ? 1
                          : c >= 0xe0 && c <= 0xef ? 2
                          : c >= 0xf0 && c <= 0xf4 ? 3
                                                   : 0;
    CHECK(continuation && continuation < value.size - i);
    unsigned next = value.data[i + 1];
    CHECK(!(c == 0xe0 && next < 0xa0) && !(c == 0xed && next >= 0xa0) &&
          !(c == 0xf0 && next < 0x90) && !(c == 0xf4 && next >= 0x90));
    for (size_t j = 1; j <= continuation; j++)
      CHECK(value.data[i + j] >= 0x80 && value.data[i + j] <= 0xbf);
    i += continuation;
  }
}
static void release_tokener(void *value) { json_tokener_free(value); }
Json parse_json(Bytes data) {
  CHECK(data.size <= LIMIT);
  utf8(data);
  int depth = 0, in_string = 0, escaped = 0;
  for (size_t i = 0; i < data.size; i++) {
    unsigned c = data.data[i];
    if (in_string) {
      if (escaped)
        escaped = 0;
      else if (c == '\\')
        escaped = 1;
      else if (c == '"')
        in_string = 0;
    } else if (c == '"')
      in_string = 1;
    else if (c == '{' || c == '[')
      CHECK(++depth <= 32);
    else if (c == '}' || c == ']')
      CHECK(--depth >= 0);
  }
  CHECK(depth == 0 && !in_string);
  struct json_tokener *tokener =
      owned(json_tokener_new_ex(33), release_tokener);
  json_tokener_set_flags(tokener,
                         JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
  Bytes encoded = json_ascii(json_keys(data, 1));
  Json value = json_tokener_parse_ex(tokener, (const char *)encoded.data,
                                     (int)encoded.size);
  if (value)
    own_json(value);
  enum json_tokener_error error = json_tokener_get_error(tokener);
  size_t parsed = json_tokener_get_parse_end(tokener);
  if (error != json_tokener_success || parsed != encoded.size) {
    fprintf(stderr, "JSON-C %s: %s, consumed %zu of %zu bytes\n",
            json_c_version(), json_tokener_error_desc(error), parsed,
            encoded.size);
    CHECK(0);
  }
  Bytes canonical = serialized(value);
  if (!same_bytes(data, canonical)) {
    size_t first = 0;
    while (first < data.size && first < canonical.size &&
           data.data[first] == canonical.data[first])
      first++;
    fprintf(stderr,
            "canonical JSON mismatch at %zu (input %zu, encoded %zu bytes)\n",
            first, data.size, canonical.size);
    CHECK(0);
  }
  return value;
}
Bytes slice(Bytes data, uint64_t offset, uint64_t length) {
  CHECK(offset <= data.size && length <= data.size - offset);
  return (Bytes){data.data + (size_t)offset, (size_t)length};
}
Bytes take(Cursor *cursor, size_t size) {
  Bytes value = slice(cursor->bytes, cursor->offset, size);
  cursor->offset += size;
  return value;
}
uint32_t read32(Cursor *cursor) {
  Bytes value = take(cursor, 4);
  return at32(value, 0);
}
uint64_t read64(Cursor *cursor) {
  Bytes value = take(cursor, 8);
  return at64(value, 0);
}
uint32_t at32(Bytes data, size_t offset) {
  Bytes value = slice(data, offset, 4);
  uint32_t result = 0;
  for (int i = 3; i >= 0; i--)
    result = (result << 8) | value.data[i];
  return result;
}
uint64_t at64(Bytes data, size_t offset) {
  Bytes value = slice(data, offset, 8);
  uint64_t result = 0;
  for (int i = 7; i >= 0; i--)
    result = (result << 8) | value.data[i];
  return result;
}
static void close_file(void *value) { fclose(value); }
Bytes read_file(const char *path) {
  FILE *file = owned(fopen(path, "rb"), close_file);
  CHECK(fseek(file, 0, SEEK_END) == 0);
  long size = ftell(file);
  CHECK(size >= 0 && (unsigned long)size <= LIMIT &&
        fseek(file, 0, SEEK_SET) == 0);
  unsigned char *data = allocate((size_t)size);
  CHECK(fread(data, 1, (size_t)size, file) == (size_t)size);
  return (Bytes){data, (size_t)size};
}
void relative_path(Bytes value) {
  CHECK(value.size && value.data[0] != '/');
  int normal = 0;
  for (size_t start = 0; start < value.size;) {
    size_t end = start;
    while (end < value.size && value.data[end] != '/')
      end++;
    CHECK(!(end - start == 2 && value.data[start] == '.' &&
            value.data[start + 1] == '.'));
    if (end > start && !(end - start == 1 && value.data[start] == '.'))
      normal = 1;
    start = end + 1;
  }
  CHECK(normal);
}
char *path_in(const char *root, Json name) {
  Bytes value = string_bytes(name);
  relative_path(value);
  CHECK(!memchr(value.data, 0, value.size));
  size_t root_length = strlen(root);
  CHECK(value.size < LIMIT - root_length - 2);
  char *joined = allocate(root_length + 1 + value.size + 1);
  memcpy(joined, root, root_length);
  joined[root_length] = '/';
  memcpy(joined + root_length + 1, value.data, value.size);
  char *resolved = owned(realpath(joined, NULL), free);
  CHECK(strncmp(resolved, root, root_length) == 0 &&
        (root_length == 1 || resolved[root_length] == '/'));
  return resolved;
}
Json hex(Bytes value) {
  static const char digits[] = "0123456789abcdef";
  CHECK(value.size <= LIMIT);
  char *result = allocate(value.size * 2 + 1);
  for (size_t i = 0; i < value.size; i++) {
    result[2 * i] = digits[value.data[i] >> 4];
    result[2 * i + 1] = digits[value.data[i] & 15];
  }
  return text((Bytes){(unsigned char *)result, value.size * 2});
}
static void release_digest(void *value) { EVP_MD_CTX_free(value); }
EVP_MD_CTX *hasher(void) {
  EVP_MD_CTX *ctx = owned(EVP_MD_CTX_new(), release_digest);
  CHECK(EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1);
  return ctx;
}
void update(EVP_MD_CTX *ctx, Bytes data) {
  CHECK(EVP_DigestUpdate(ctx, data.data, data.size) == 1);
}
Json finish(EVP_MD_CTX *ctx) {
  unsigned char data[32];
  unsigned length = 0;
  CHECK(EVP_DigestFinal_ex(ctx, data, &length) == 1 && length == 32);
  return hex((Bytes){data, 32});
}
Json hash(Bytes data) {
  EVP_MD_CTX *ctx = hasher();
  update(ctx, data);
  return finish(ctx);
}
