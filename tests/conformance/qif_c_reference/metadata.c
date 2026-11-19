#include "qif.h"
#include <string.h>

static Json value(Cursor *, unsigned);
static Bytes qstring(Cursor *cursor) {
  uint64_t length = read64(cursor);
  CHECK(length <= 16 * 1024 * 1024);
  Bytes result = take(cursor, (size_t)length);
  utf8(result);
  return result;
}
static int before(Bytes left, Bytes right) {
  size_t length = left.size < right.size ? left.size : right.size;
  int order = memcmp(left.data, right.data, length);
  return order < 0 || (order == 0 && left.size < right.size);
}
static Json value(Cursor *cursor, unsigned depth) {
  CHECK(depth <= 32);
  uint32_t tag = read32(cursor);
  const char *name = NULL;
  Json result = NULL;
  switch (tag) {
  case 4:
    name = "U32";
    result = number(read32(cursor));
    break;
  case 5: {
    name = "I32";
    uint32_t bits = read32(cursor);
    int32_t signed_value;
    memcpy(&signed_value, &bits, 4);
    result = own_json(json_object_new_int64(signed_value));
    break;
  }
  case 10:
    name = "U64";
    result = number(read64(cursor));
    break;
  case 11: {
    name = "I64";
    uint64_t bits = read64(cursor);
    int64_t signed_value;
    memcpy(&signed_value, &bits, 8);
    result = own_json(json_object_new_int64(signed_value));
    break;
  }
  case 6:
  case 12: {
    name = tag == 6 ? "F32" : "F64";
    Bytes raw = take(cursor, tag == 6 ? 4 : 8);
    if (tag == 6)
      CHECK((at32(raw, 0) & 0x7f800000) != 0x7f800000);
    else
      CHECK((at64(raw, 0) & UINT64_C(0x7ff0000000000000)) !=
            UINT64_C(0x7ff0000000000000));
    result = hex(raw);
    break;
  }
  case 7: {
    name = "Bool";
    unsigned bit = take(cursor, 1).data[0];
    CHECK(bit <= 1);
    result = own_json(json_object_new_boolean(bit));
    break;
  }
  case 8:
    name = "String";
    result = text(qstring(cursor));
    break;
  case 9:
  case 13: {
    name = tag == 9 ? "Array" : "Map";
    uint64_t count = read64(cursor);
    CHECK(count <= 1048576);
    result = tag == 9 ? array() : object();
    Bytes previous = {NULL, 0};
    Json element_type = NULL;
    for (uint64_t i = 0; i < count; i++) {
      if (tag == 13) {
        Bytes key = qstring(cursor);
        CHECK(key.size && (!i || before(previous, key)));
        put_key(result, key, value(cursor, depth + 1));
        previous = key;
      } else {
        Json item = value(cursor, depth + 1), type = field(item, "type");
        CHECK(!i || same(type, element_type));
        element_type = type;
        push(result, item);
      }
    }
    break;
  }
  default:
    CHECK(0);
  }
  Json wrapped = object();
  put(wrapped, "type", string(name));
  put(wrapped, "value", result);
  return wrapped;
}
Json qmb(Bytes data) {
  Cursor cursor = {data, 0};
  Json result = value(&cursor, 0);
  CHECK(same(field(result, "type"), string("Map")) &&
        cursor.offset == data.size);
  Json dimension_order = NULL;
  if (json_object_object_get_ex(field(result, "value"),
                                "quantinfer.qif.ggml_dimension_order", &dimension_order))
    CHECK(same(field(dimension_order, "type"), string("Bool")));
  return field(result, "value");
}
void finite_payload(uint32_t dtype, Bytes data) {
  static const size_t widths[] = {0, 4, 2, 2, 18, 144, 176, 210, 34};
  CHECK(dtype >= 1 && dtype <= 8 && data.size % widths[dtype] == 0);
  for (size_t offset = 0; offset < data.size; offset += widths[dtype]) {
    if (dtype == 1)
      CHECK((at32(data, offset) & 0x7f800000) != 0x7f800000);
    else {
      size_t scale = offset + (dtype == 7 ? 208 : 0);
      Bytes raw = slice(data, scale, 2);
      unsigned bits = raw.data[0] | ((unsigned)raw.data[1] << 8);
      unsigned exponent = dtype == 3 ? 0x7f80 : 0x7c00;
      CHECK((bits & exponent) != exponent);
      if (dtype == 5 || dtype == 6) {
        raw = slice(data, offset + 2, 2);
        bits = raw.data[0] | ((unsigned)raw.data[1] << 8);
        CHECK((bits & 0x7c00) != 0x7c00);
      }
    }
  }
}
