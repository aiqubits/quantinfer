/* JSON-C keys are NUL-terminated. Encode exceptional keys internally and
 * restore their exact UTF-8 bytes on canonical serialization. Prefix-bearing
 * keys are encoded too, making this mapping injective, without sentinels or
 * probabilistic collision assumptions. Key mapping preserves string values;
 * the ASCII parser buffer below preserves decoded values as well. */
#include "qif.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char prefix[] = "~qif-key:";
typedef struct {
  unsigned char *data;
  size_t size, capacity;
} Buffer;
static void release_buffer(void *value) {
  Buffer *buffer = value;
  free(buffer->data);
  free(buffer);
}
static Buffer *buffer(void) {
  return owned(calloc(1, sizeof(Buffer)), release_buffer);
}
static void append(Buffer *out, Bytes data) {
  if (!data.size)
    return;
  CHECK(out->size <= LIMIT * 4u && data.size <= LIMIT * 4u - out->size);
  size_t end = out->size + data.size;
  if (end > out->capacity) {
    size_t capacity = out->capacity ? out->capacity : 256;
    while (capacity < end) {
      CHECK(capacity <= LIMIT * 2u);
      capacity *= 2;
    }
    void *replacement = realloc(out->data, capacity);
    CHECK(replacement);
    out->data = replacement;
    out->capacity = capacity;
  }
  memcpy(out->data + out->size, data.data, data.size);
  out->size = end;
}
Bytes json_ascii(Bytes input) {
  // json-c 0.18 strict mode treats signed non-ASCII char values as controls.
  // Keep strict parsing, prevalidate UTF-8 independently, and provide an
  // equivalent ASCII parser buffer. Canonical checks still use original bytes.
  utf8(input);
  Buffer *out = buffer();
  size_t copied = 0;
  for (size_t i = 0; i < input.size;) {
    unsigned c = input.data[i];
    if (c < 128) {
      i++;
      continue;
    }
    size_t width = c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    unsigned codepoint = c & ((1u << (7 - width)) - 1);
    for (size_t j = 1; j < width; j++)
      codepoint = (codepoint << 6) | (input.data[i + j] & 63);
    append(out, slice(input, copied, i - copied));
    char escaped[13];
    int length;
    if (codepoint <= 0xffff)
      length = snprintf(escaped, sizeof(escaped), "\\u%04x", codepoint);
    else {
      codepoint -= 0x10000;
      length =
          snprintf(escaped, sizeof(escaped), "\\u%04x\\u%04x",
                   0xd800 + (codepoint >> 10), 0xdc00 + (codepoint & 1023));
    }
    CHECK(length == 6 || length == 12);
    append(out, (Bytes){(unsigned char *)escaped, (size_t)length});
    i += width;
    copied = i;
  }
  if (!copied)
    return input;
  append(out, slice(input, copied, input.size - copied));
  return (Bytes){out->data, out->size};
}
static int exceptional(Bytes key) {
  return memchr(key.data, 0, key.size) != NULL ||
         (key.size >= sizeof(prefix) - 1 &&
          memcmp(key.data, prefix, sizeof(prefix) - 1) == 0);
}
static Json encoded(Bytes key) {
  if (!exceptional(key))
    return text(key);
  Bytes digits = string_bytes(hex(key));
  size_t length = sizeof(prefix) - 1 + digits.size;
  unsigned char *out = allocate(length);
  memcpy(out, prefix, sizeof(prefix) - 1);
  memcpy(out + sizeof(prefix) - 1, digits.data, digits.size);
  return text((Bytes){out, length});
}
static unsigned nibble(unsigned c) {
  CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
  return c <= '9' ? c - '0' : c - 'a' + 10;
}
static Json decoded(Bytes key) {
  if (key.size < sizeof(prefix) - 1 ||
      memcmp(key.data, prefix, sizeof(prefix) - 1) != 0)
    return text(key);
  Bytes digits =
      slice(key, sizeof(prefix) - 1, key.size - (sizeof(prefix) - 1));
  CHECK(digits.size % 2 == 0);
  unsigned char *out = allocate(digits.size / 2);
  for (size_t i = 0; i < digits.size; i += 2)
    out[i / 2] = (unsigned char)((nibble(digits.data[i]) << 4) |
                                 nibble(digits.data[i + 1]));
  Bytes original = {out, digits.size / 2};
  CHECK(exceptional(original));
  utf8(original);
  return text(original);
}
Json key_text(const char *key) {
  return decoded((Bytes){(const unsigned char *)key, strlen(key)});
}
void put_key(Json parent, Bytes key, Json value) {
  Json stored = encoded(key);
  Bytes data = string_bytes(stored);
  char *name = allocate(data.size + 1);
  memcpy(name, data.data, data.size);
  put(parent, name, value);
}
static void release_tokener(void *value) { json_tokener_free(value); }
Bytes json_keys(Bytes input, int encode) {
  Buffer *out = buffer();
  size_t copied = 0;
  for (size_t start = 0; start < input.size;) {
    if (input.data[start] != '"') {
      start++;
      continue;
    }
    size_t end = start + 1;
    int escaped = 0;
    while (end < input.size) {
      unsigned c = input.data[end++];
      if (escaped)
        escaped = 0;
      else if (c == '\\')
        escaped = 1;
      else if (c == '"')
        break;
    }
    CHECK(end <= input.size && end > start + 1 && input.data[end - 1] == '"');
    size_t next = end;
    while (next < input.size &&
           (input.data[next] == ' ' || input.data[next] == '\n' ||
            input.data[next] == '\t' || input.data[next] == '\r'))
      next++;
    if (next < input.size && input.data[next] == ':') {
      // Standalone string parsing retains embedded NULs with a length.
      struct json_tokener *tokener = owned(json_tokener_new(), release_tokener);
      json_tokener_set_flags(tokener,
                             JSON_TOKENER_STRICT | JSON_TOKENER_VALIDATE_UTF8);
      Bytes literal_input = json_ascii(slice(input, start, end - start));
      Json key = json_tokener_parse_ex(
          tokener, (const char *)literal_input.data, (int)literal_input.size);
      if (key)
        own_json(key);
      CHECK(key && json_tokener_get_error(tokener) == json_tokener_success &&
            json_object_is_type(key, json_type_string));
      Json mapped =
          encode ? encoded(string_bytes(key)) : decoded(string_bytes(key));
      if (!same(key, mapped)) {
        append(out, slice(input, copied, start - copied));
        const char *literal = json_object_to_json_string_ext(
            mapped, JSON_C_TO_STRING_PLAIN | JSON_C_TO_STRING_NOSLASHESCAPE);
        CHECK(literal);
        append(out, (Bytes){(const unsigned char *)literal, strlen(literal)});
        copied = end;
      }
    }
    start = end;
  }
  if (!copied)
    return input;
  append(out, slice(input, copied, input.size - copied));
  return (Bytes){out->data, out->size};
}
