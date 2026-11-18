#ifndef QIF_REFERENCE_H
#define QIF_REFERENCE_H
#define _XOPEN_SOURCE 700
#include <json-c/json.h>
#include <limits.h>
#include <openssl/evp.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) == 8 && sizeof(long) >= 8,
               "reference CLI requires a 64-bit POSIX host with 8-bit bytes");

#define LIMIT (64u * 1024u * 1024u)
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c))                                                                  \
      fail(__FILE__, __LINE__);                                                \
  } while (0)
typedef struct {
  const unsigned char *data;
  size_t size;
} Bytes;
typedef struct {
  Bytes bytes;
  size_t offset;
} Cursor;
typedef struct json_object *Json;
typedef struct {
  Json name;
  Bytes data;
} Shard;
typedef struct {
  Shard *items;
  size_t count;
  int package;
} Shards;
extern jmp_buf qif_error;
_Noreturn void fail(const char *, int);
void *owned(void *, void (*)(void *));
void *allocate(size_t);
void cleanup(void);
Json own_json(Json);
Json object(void);
Json array(void);
Json text(Bytes);
Json string(const char *);
Json key_text(const char *);
void put_key(Json, Bytes, Json);
Json number(uint64_t);
void put(Json, const char *, Json);
void push(Json, Json);
Json field(Json, const char *);
uint64_t integer(Json);
Bytes string_bytes(Json);
int same_bytes(Bytes, Bytes);
int same(Json, Json);
void keys(Json, const char *);
Bytes serialized(Json);
Json parse_json(Bytes);
Bytes json_keys(Bytes, int);
Bytes json_ascii(Bytes);
void utf8(Bytes);
Bytes slice(Bytes, uint64_t, uint64_t);
Bytes take(Cursor *, size_t);
uint32_t read32(Cursor *);
uint64_t read64(Cursor *);
uint32_t at32(Bytes, size_t);
uint64_t at64(Bytes, size_t);
Bytes read_file(const char *);
char *path_in(const char *, Json);
void relative_path(Bytes);
Json hex(Bytes);
Json hash(Bytes);
EVP_MD_CTX *hasher(void);
void update(EVP_MD_CTX *, Bytes);
Json finish(EVP_MD_CTX *);
Json qmb(Bytes);
Bytes decode(uint32_t, Bytes, size_t);
void finite_payload(uint32_t, Bytes);
void validate_model(Json);
void validate_descriptor(Json, Json);
Json parse_index(Bytes, Json, const Shards *);
Json report(const char *, Json);
#endif
