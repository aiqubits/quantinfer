#include "qif.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void release_pkey(void *value) { EVP_PKEY_free(value); }
static void release_verify(void *value) { EVP_MD_CTX_free(value); }
static int little_less(const unsigned char *left, const unsigned char *right) {
  for (int i = 31; i >= 0; i--)
    if (left[i] != right[i])
      return left[i] < right[i];
  return 0;
}
static void signature_point(Bytes point) {
  unsigned char y[32], prime[32];
  memcpy(y, point.data, 32);
  y[31] &= 127;
  memset(prime, 255, 32);
  prime[0] = 237;
  prime[31] = 127;
  CHECK(little_less(y, prime));
  unsigned tail = 0;
  for (size_t i = 1; i < 32; i++)
    tail |= y[i];
  CHECK(tail || y[0] > 1);
  prime[0] = 236;
  CHECK(memcmp(y, prime, 32) != 0);
  Json encoded = hex((Bytes){y, 32});
  CHECK(!same(
      encoded,
      string(
          "26e8958fc2b227b045c3f489f2ef98f0d5dfac05d3c63339b13802886d53fc05")));
  CHECK(!same(
      encoded,
      string(
          "c7176a703d4dd84fba3c0b760d10670f2a2053fa2c39ccc64ec7fd7792ac037a")));
}
static Json signature(Bytes root, Bytes envelope, Json trusted) {
  if (envelope.size == 0) {
    CHECK(json_object_array_length(trusted) == 0);
    return NULL;
  }
  CHECK(envelope.size == 108 &&
        same_bytes(slice(envelope, 0, 4),
                   (Bytes){(const unsigned char *)"QSIG", 4}) &&
        at32(envelope, 4) == 1 && at32(envelope, 8) == 1);
  Bytes key = slice(envelope, 12, 32), sig = slice(envelope, 44, 64);
  signature_point(key);
  signature_point(slice(sig, 0, 32));
  const unsigned char order[32] = {
      0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58, 0xd6, 0x9c, 0xf7,
      0xa2, 0xde, 0xf9, 0xde, 0x14, 0,    0,    0,    0,    0,    0,
      0,    0,    0,    0,    0,    0,    0,    0,    0,    0x10};
  CHECK(little_less(sig.data + 32, order));
  EVP_PKEY *public_key = owned(
      EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, NULL, key.data, key.size),
      release_pkey);
  EVP_MD_CTX *ctx = owned(EVP_MD_CTX_new(), release_verify);
  CHECK(EVP_DigestVerifyInit(ctx, NULL, NULL, NULL, public_key) == 1 &&
        EVP_DigestVerify(ctx, sig.data, sig.size, root.data, root.size) == 1);
  Json public_hex = hex(key);
  size_t count = json_object_array_length(trusted);
  int accepted = count == 0;
  for (size_t i = 0; i < count; i++)
    if (same(public_hex, json_object_array_get_idx(trusted, i)))
      accepted = 1;
  CHECK(accepted);
  Json value = object();
  put(value, "public_key", public_hex);
  put(value, "verified", own_json(json_object_new_boolean(1)));
  put(value, "publisher_trust", string(count ? "trusted" : "not_evaluated"));
  return value;
}
static Json single(const char *path, Json trusted) {
  Bytes data = read_file(path);
  CHECK(data.size >= 144 &&
        same_bytes(slice(data, 0, 4),
                   (Bytes){(const unsigned char *)"QIF\0", 4}));
  CHECK(at32(data, 4) == 2 && at32(data, 8) == 0 && at32(data, 12) == 144 &&
        at32(data, 16) == 1 && at32(data, 20) == 1 && at64(data, 136) == 0);
  uint64_t offsets[5] = {0}, lengths[5] = {0};
  Bytes sections[4];
  for (size_t i = 0; i < 4; i++) {
    offsets[i] = at64(data, 24 + i * 16);
    lengths[i] = at64(data, 32 + i * 16);
    CHECK(offsets[i] >= 144 && offsets[i] % 4096 == 0);
    sections[i] = slice(data, offsets[i], lengths[i]);
  }
  CHECK(offsets[3] + lengths[3] == data.size);
  offsets[4] = at64(data, 120);
  lengths[4] = at64(data, 128);
  Bytes envelope = {NULL, 0};
  if (lengths[4]) {
    CHECK(lengths[4] == 108 && offsets[4] >= 144 && offsets[4] % 4096 == 0);
    envelope = slice(data, offsets[4], lengths[4]);
  } else
    CHECK(offsets[4] == 0);
  for (size_t i = 0; i < 5; i++)
    for (size_t j = 0; j < i; j++)
      if (lengths[i] && lengths[j])
        CHECK(offsets[i] >= offsets[j] + lengths[j] ||
              offsets[j] >= offsets[i] + lengths[i]);
  EVP_MD_CTX *ctx = hasher();
  for (size_t i = 0; i < 3; i++)
    update(ctx, sections[i]);
  Json root = hex(slice(data, 88, 32));
  CHECK(same(finish(ctx), root));
  Json model = parse_json(sections[0]);
  validate_model(model);
  Json verified_signature = signature(slice(data, 88, 32), envelope, trusted);
  Shard payload = {NULL, sections[3]};
  Shards shards = {&payload, 1, 0};
  Json result = parse_index(sections[2], model, &shards);
  put(result, "profile", string("SINGLE"));
  put(result, "root_hash", root);
  put(result, "model", model);
  put(result, "metadata", qmb(sections[1]));
  put(result, "metadata_sha256", hash(sections[1]));
  put(result, "index_sha256", hash(sections[2]));
  put(result, "signature", verified_signature);
  return result;
}
static int contains(Json list, Json value) {
  CHECK(json_object_is_type(list, json_type_array));
  for (size_t i = 0; i < json_object_array_length(list); i++)
    if (same(json_object_array_get_idx(list, i), value))
      return 1;
  return 0;
}
static Json package(const char *path, Json trusted) {
  CHECK(json_object_array_length(trusted) == 0);
  char *root = owned(realpath(path, NULL), free);
  Bytes package_bytes = read_file(path_in(root, string("qif.json")));
  Json document = parse_json(package_bytes);
  keys(document, "schema_version profile model metadata_path metadata_sha256 "
                 "index_path index_sha256 shards root_hash");
  CHECK(integer(field(document, "schema_version")) == 1 &&
        (same(field(document, "profile"), string("QIF-PACKAGE")) ||
         same(field(document, "profile"), string("QIF2-PACKAGE"))));
  Json model = field(document, "model");
  validate_model(model);
  Bytes metadata = read_file(path_in(root, field(document, "metadata_path"))),
        index = read_file(path_in(root, field(document, "index_path")));
  CHECK(same(hash(metadata), field(document, "metadata_sha256")) &&
        same(hash(index), field(document, "index_sha256")));
  Json root_hash = text(string_bytes(field(document, "root_hash")));
  put(document, "root_hash", string(""));
  EVP_MD_CTX *ctx = hasher();
  update(ctx, serialized(document));
  update(ctx, metadata);
  update(ctx, index);
  Json entries = field(document, "shards");
  CHECK(json_object_is_type(entries, json_type_array));
  size_t count = json_object_array_length(entries);
  CHECK(count <= LIMIT / sizeof(Shard));
  for (size_t i = 0; i < count; i++) {
    Json entry = json_object_array_get_idx(entries, i);
    keys(entry, "path size sha256 mesh_axis rank coordinate tensors");
    update(ctx, string_bytes(field(entry, "sha256")));
  }
  CHECK(same(finish(ctx), root_hash));
  put(document, "root_hash", root_hash);
  Shard *items = allocate(count * sizeof(*items));
  uint64_t budget = package_bytes.size + metadata.size + index.size;
  CHECK(budget <= LIMIT);
  for (size_t i = 0; i < count; i++) {
    Json entry = json_object_array_get_idx(entries, i),
         name = field(entry, "path");
    for (size_t j = 0; j < i; j++)
      CHECK(!same(name, items[j].name));
    uint64_t size = integer(field(entry, "size"));
    CHECK(size <= LIMIT - budget);
    budget += size;
    Bytes data = read_file(path_in(root, name));
    CHECK(data.size == size && same(hash(data), field(entry, "sha256")));
    CHECK(integer(field(entry, "mesh_axis")) <= UINT32_MAX);
    uint64_t rank = integer(field(entry, "rank"));
    Json coordinate = field(entry, "coordinate");
    CHECK(rank > 0 && rank <= UINT32_MAX &&
          json_object_is_type(coordinate, json_type_array) &&
          json_object_array_length(coordinate) == rank);
    for (size_t j = 0; j < rank; j++)
      CHECK(integer(json_object_array_get_idx(coordinate, j)) <= UINT32_MAX);
    Json names = field(entry, "tensors");
    CHECK(json_object_is_type(names, json_type_array));
    for (size_t j = 0; j < json_object_array_length(names); j++)
      CHECK(string_bytes(json_object_array_get_idx(names, j)).size <=
            16 * 1024 * 1024);
    items[i] = (Shard){name, data};
  }
  Shards shards = {items, count, 1};
  Json result = parse_index(index, model, &shards);
  Json tensors = field(result, "tensors"), chunks = field(result, "chunks");
  for (size_t i = 0; i < count; i++) {
    Json entry = json_object_array_get_idx(entries, i),
         expected = field(entry, "tensors"), actual = array();
    for (size_t j = 0; j < json_object_array_length(tensors); j++) {
      Json tensor = json_object_array_get_idx(tensors, j), physical = tensor,
           alias = field(tensor, "alias_of");
      if (alias)
        physical = json_object_array_get_idx(tensors, (size_t)integer(alias));
      uint64_t first = integer(field(physical, "chunk_start")),
               length = integer(field(physical, "chunk_count"));
      int present = 0;
      for (uint64_t k = first; k < first + length; k++) {
        Json chunk = json_object_array_get_idx(chunks, (size_t)k);
        char name[40];
        CHECK(snprintf(name, sizeof(name), "shard-%05u.qifd",
                       (unsigned)integer(field(chunk, "shard_id"))) > 0);
        if (same(items[i].name, string(name)))
          present = 1;
      }
      if (present) {
        Json name = field(tensor, "name");
        CHECK(contains(expected, name));
        push(actual, name);
      }
    }
    for (size_t j = 0; j < json_object_array_length(expected); j++)
      CHECK(contains(actual, json_object_array_get_idx(expected, j)));
  }
  put(result, "profile", string("PACKAGE"));
  put(result, "root_hash", root_hash);
  put(result, "model", model);
  put(result, "metadata", qmb(metadata));
  put(result, "metadata_sha256", hash(metadata));
  put(result, "index_sha256", hash(index));
  put(result, "signature", NULL);
  return result;
}
Json report(const char *path, Json trusted) {
  struct stat info;
  CHECK(stat(path, &info) == 0);
  return S_ISDIR(info.st_mode) ? package(path, trusted) : single(path, trusted);
}
