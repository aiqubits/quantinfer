#include "qif.h"
#include <lz4.h>
#include <string.h>
#include <zstd.h>

static void release_zstd(void *value) { ZSTD_freeDCtx(value); }
Bytes decode(uint32_t codec, Bytes stored, size_t expected) {
  CHECK(expected <= LIMIT);
  if (codec == 0) {
    CHECK(stored.size == expected);
    return stored;
  }
  unsigned char *output = allocate(expected + 1);
  if (codec == 2) {
    CHECK(at32(stored, 0) == expected && stored.size >= 4 &&
          stored.size - 4 <= INT32_MAX);
    int length =
        LZ4_decompress_safe((const char *)stored.data + 4, (char *)output,
                            (int)stored.size - 4, (int)expected);
    CHECK(length >= 0 && (size_t)length == expected);
  } else {
    CHECK(codec == 1);
    ZSTD_DCtx *ctx = owned(ZSTD_createDCtx(), release_zstd);
    CHECK(!ZSTD_isError(ZSTD_DCtx_setParameter(ctx, ZSTD_d_windowLogMax, 27)));
    ZSTD_inBuffer in = {stored.data, stored.size, 0};
    ZSTD_outBuffer out = {output, expected + 1, 0};
    for (;;) {
      size_t old_in = in.pos, old_out = out.pos,
             status = ZSTD_decompressStream(ctx, &out, &in);
      CHECK(!ZSTD_isError(status) && out.pos <= expected);
      if (status == 0 && in.pos == stored.size)
        break;
      CHECK(old_in != in.pos || old_out != out.pos);
    }
    CHECK(out.pos == expected);
  }
  return (Bytes){output, expected};
}
