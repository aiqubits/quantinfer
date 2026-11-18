#include "qif.h"
#include <stdio.h>
#include <string.h>

typedef struct {
  Json report;
  uint64_t raw_offset, raw, offset, stored;
  uint32_t shard, codec;
  Bytes checksum;
  int referenced;
} Chunk;
typedef struct {
  Json report;
  uint64_t alias, raw, stored, offset;
  uint32_t start, count, codec;
  Bytes payload;
} Tensor;
static Json name_at(Bytes data, uint32_t offset) {
  Cursor cursor = {data, offset};
  uint32_t length = read32(&cursor);
  CHECK(length <= 16 * 1024 * 1024);
  Bytes value = take(&cursor, length);
  utf8(value);
  return text(value);
}
static Json shape_at(Bytes data, uint32_t offset) {
  Cursor cursor = {data, offset};
  uint32_t rank = read32(&cursor);
  CHECK(rank <= 32);
  Json result = array();
  for (uint32_t i = 0; i < rank; i++) {
    uint64_t dimension = read64(&cursor);
    CHECK(dimension);
    push(result, number(dimension));
  }
  return result;
}
static Bytes shard_data(const Shards *shards, uint32_t id) {
  if (!shards->package) {
    CHECK(id == 0 && shards->count == 1);
    return shards->items[0].data;
  }
  char name[40];
  CHECK(snprintf(name, sizeof(name), "shard-%05u.qifd", id) > 0);
  for (size_t i = 0; i < shards->count; i++)
    if (same_bytes(string_bytes(shards->items[i].name),
                   (Bytes){(unsigned char *)name, strlen(name)}))
      return shards->items[i].data;
  CHECK(0);
}
Json parse_index(Bytes data, Json model, const Shards *shards) {
  Cursor cursor = {data, 0};
  CHECK(
      same_bytes(take(&cursor, 4), (Bytes){(const unsigned char *)"QIDX", 4}) &&
      read32(&cursor) == 1);
  uint32_t tensor_count = read32(&cursor), chunk_count = read32(&cursor);
  Json manifest = field(model, "tensors");
  CHECK(tensor_count <= 1048576 && chunk_count <= 4194304 &&
        tensor_count == json_object_array_length(manifest));
  uint64_t lengths[4];
  for (size_t i = 0; i < 4; i++)
    lengths[i] = read64(&cursor);
  CHECK(lengths[2] == (uint64_t)tensor_count * 120 &&
        lengths[3] == (uint64_t)chunk_count * 72);
  Bytes strings = take(&cursor, (size_t)lengths[0]),
        shapes = take(&cursor, (size_t)lengths[1]),
        records = take(&cursor, (size_t)lengths[2]),
        chunk_bytes = take(&cursor, (size_t)lengths[3]);
  CHECK(cursor.offset == data.size);
  Chunk *chunks = allocate((size_t)chunk_count * sizeof(*chunks));
  Tensor *tensors = allocate((size_t)tensor_count * sizeof(*tensors));
  Json chunk_reports = array(), tensor_reports = array();
  for (uint32_t i = 0; i < chunk_count; i++) {
    Cursor entry = {slice(chunk_bytes, (uint64_t)i * 72, 72), 0};
    Chunk *chunk = &chunks[i];
    chunk->report = object();
    chunk->shard = read32(&entry);
    chunk->codec = read32(&entry);
    chunk->raw_offset = read64(&entry);
    chunk->raw = read64(&entry);
    chunk->offset = read64(&entry);
    chunk->stored = read64(&entry);
    chunk->checksum = take(&entry, 32);
    CHECK(chunk->codec <= 2 && chunk->raw && chunk->stored &&
          chunk->raw_offset <= UINT64_MAX - chunk->raw &&
          chunk->offset <= UINT64_MAX - chunk->stored &&
          chunk->offset % 4096 == 0);
    CHECK(chunk->codec == 0 ? chunk->raw == chunk->stored
                            : chunk->raw <= 128 * 1024 * 1024 &&
                                  chunk->stored <= 129 * 1024 * 1024);
    Bytes stored =
        slice(shard_data(shards, chunk->shard), chunk->offset, chunk->stored);
    CHECK(same(hash(stored), hex(chunk->checksum)));
    for (uint32_t j = 0; j < i; j++)
      if (chunks[j].shard == chunk->shard)
        CHECK(chunk->offset >= chunks[j].offset + chunks[j].stored ||
              chunks[j].offset >= chunk->offset + chunk->stored);
    put(chunk->report, "shard_id", number(chunk->shard));
    put(chunk->report, "codec", number(chunk->codec));
    put(chunk->report, "raw_offset", number(chunk->raw_offset));
    put(chunk->report, "raw_length", number(chunk->raw));
    put(chunk->report, "stored_offset", number(chunk->offset));
    put(chunk->report, "stored_length", number(chunk->stored));
    put(chunk->report, "checksum_sha256", hex(chunk->checksum));
    push(chunk_reports, chunk->report);
  }
  size_t decoded_total = 0;
  for (uint32_t i = 0; i < tensor_count; i++) {
    Cursor entry = {slice(records, (uint64_t)i * 120, 120), 0};
    Tensor *tensor = &tensors[i];
    Json value = object();
    tensor->report = value;
    CHECK(read64(&entry) == i);
    put(value, "tensor_id", number(i));
    Json name = name_at(strings, read32(&entry)),
         shape = shape_at(shapes, read32(&entry));
    put(value, "name", name);
    put(value, "shape", shape);
    const char *encoding_keys[] = {"dtype", "physical_encoding",
                                   "quantization_scheme", "layout", "codec"};
    for (size_t j = 0; j < 5; j++)
      put(value, encoding_keys[j], number(read32(&entry)));
    tensor->codec = (uint32_t)integer(field(value, "codec"));
    CHECK(read32(&entry) == 0);
    tensor->start = read32(&entry);
    tensor->count = read32(&entry);
    CHECK(tensor->start <= chunk_count &&
          tensor->count <= chunk_count - tensor->start);
    tensor->raw = read64(&entry);
    tensor->stored = read64(&entry);
    tensor->offset = read64(&entry);
    put(value, "chunk_start", number(tensor->start));
    put(value, "chunk_count", number(tensor->count));
    put(value, "raw_length", number(tensor->raw));
    put(value, "stored_length", number(tensor->stored));
    put(value, "data_offset", number(tensor->offset));
    put(value, "checksum_sha256", hex(take(&entry, 32)));
    tensor->alias = read64(&entry);
    CHECK(read32(&entry) == 0 && read32(&entry) == 0);
    put(value, "alias_of",
        tensor->alias == UINT64_MAX ? NULL : number(tensor->alias));
    Json manifest_tensor = NULL;
    for (uint32_t j = 0; j < tensor_count; j++) {
      Json candidate = json_object_array_get_idx(manifest, j);
      if (same(field(candidate, "name"), name))
        manifest_tensor = candidate;
    }
    CHECK(manifest_tensor);
    validate_descriptor(value, manifest_tensor);
    CHECK(tensor->raw <= LIMIT - decoded_total);
    decoded_total += (size_t)tensor->raw;
    for (uint32_t j = 0; j < i; j++)
      CHECK(!same(field(tensors[j].report, "name"), name));
    if (tensor->alias != UINT64_MAX) {
      CHECK(tensor->alias < i && tensor->count == 0);
      Tensor *target = &tensors[tensor->alias];
      CHECK(target->alias == UINT64_MAX && target->count);
      const char *alias_keys[] = {"shape",
                                  "dtype",
                                  "physical_encoding",
                                  "quantization_scheme",
                                  "layout",
                                  "codec",
                                  "raw_length",
                                  "stored_length",
                                  "checksum_sha256",
                                  "data_offset"};
      for (size_t j = 0; j < 10; j++)
        CHECK(same(field(value, alias_keys[j]),
                   field(target->report, alias_keys[j])));
      tensor->payload = target->payload;
      put(value, "payload_hex", field(target->report, "payload_hex"));
    } else {
      CHECK(tensor->count);
      unsigned char *output = allocate((size_t)tensor->raw);
      uint64_t raw_sum = 0, stored_sum = 0;
      for (uint32_t j = tensor->start; j < tensor->start + tensor->count; j++) {
        Chunk *chunk = &chunks[j];
        CHECK(!chunk->referenced);
        chunk->referenced = 1;
        CHECK(chunk->codec == tensor->codec && chunk->raw_offset == raw_sum);
        if (j == tensor->start)
          CHECK(chunk->offset == tensor->offset);
        for (uint32_t k = j; k > tensor->start;) {
          k--;
          if (chunks[k].shard == chunk->shard) {
            uint64_t end = chunks[k].offset + chunks[k].stored;
            CHECK(end <= UINT64_MAX - 4095 &&
                  chunk->offset == (end + 4095) / 4096 * 4096);
            break;
          }
        }
        CHECK(raw_sum <= tensor->raw && chunk->raw <= tensor->raw - raw_sum &&
              stored_sum <= UINT64_MAX - chunk->stored);
        Bytes stored = slice(shard_data(shards, chunk->shard), chunk->offset,
                             chunk->stored),
              decoded = decode(chunk->codec, stored, (size_t)chunk->raw);
        memcpy(output + (size_t)raw_sum, decoded.data, decoded.size);
        raw_sum += chunk->raw;
        stored_sum += chunk->stored;
      }
      CHECK(raw_sum == tensor->raw && stored_sum == tensor->stored);
      tensor->payload = (Bytes){output, (size_t)tensor->raw};
      CHECK(same(hash(tensor->payload), field(value, "checksum_sha256")));
      finite_payload((uint32_t)integer(field(value, "dtype")), tensor->payload);
      put(value, "payload_hex", hex(tensor->payload));
    }
    push(tensor_reports, value);
  }
  for (uint32_t i = 0; i < chunk_count; i++)
    CHECK(chunks[i].referenced);
  Json result = object();
  put(result, "tensors", tensor_reports);
  put(result, "chunks", chunk_reports);
  return result;
}
