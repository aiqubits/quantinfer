/* Independent unsigned, uncompressed F32 golden-profile reader. OpenSSL
 * supplies SHA-256. This tool is not a general ModelManifest/QMB validator. */
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>

#define REQUIRE(c) do { if (!(c)) { fprintf(stderr, "invalid/unsupported QIF golden input at line %d\n", __LINE__); return 1; } } while (0)

static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t u64(const unsigned char *p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
    return value;
}
static const unsigned char *region(const unsigned char *data, size_t size, uint64_t start, uint64_t length) {
    if (start > size || length > size - start) return NULL;
    return data + (size_t)start;
}
static int digest(const unsigned char *data, size_t size, unsigned char out[32]) {
    unsigned int length = 0;
    return EVP_Digest(data, size, out, &length, EVP_sha256(), NULL) == 1 && length == 32;
}
static void hex(const unsigned char *bytes, size_t length) {
    for (size_t i = 0; i < length; ++i) printf("%02x", bytes[i]);
}
static void hash_field(const char *name, const unsigned char *value) {
    printf(",\"%s\":\"", name); hex(value, 32); printf("\"");
}

static int inspect(const unsigned char *data, size_t size, int json) {
    REQUIRE(size >= 144 && memcmp(data, "QIF\0", 4) == 0);
    REQUIRE(u32(data+4) == 2 && u32(data+8) == 0 && u32(data+12) == 144);
    REQUIRE(u32(data+16) == 1 && u32(data+20) == 1);
    REQUIRE(u64(data+120) == 0 && u64(data+128) == 0 && u64(data+136) == 0);
    uint64_t offsets[4], lengths[4];
    const unsigned char *sections[4];
    unsigned char hashes[3][32], root_hash[32], payload_hash[32];
    for (size_t i = 0; i < 4; ++i) {
        offsets[i] = u64(data+24+i*16);
        lengths[i] = u64(data+32+i*16);
        REQUIRE(offsets[i] >= 144 && offsets[i] % 4096 == 0);
        sections[i] = region(data, size, offsets[i], lengths[i]);
        REQUIRE(sections[i] != NULL);
        for (size_t j = 0; j < i; ++j)
            REQUIRE(offsets[i] >= offsets[j]+lengths[j] || offsets[j] >= offsets[i]+lengths[i]);
        if (i < 3) REQUIRE(digest(sections[i], (size_t)lengths[i], hashes[i]));
    }
    REQUIRE(offsets[3]+lengths[3] == size);
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    REQUIRE(ctx != NULL);
    int ok = EVP_DigestInit_ex(ctx, EVP_sha256(), NULL) == 1;
    for (size_t i = 0; i < 3 && ok; ++i) ok = EVP_DigestUpdate(ctx, sections[i], (size_t)lengths[i]) == 1;
    unsigned int hash_length = 0;
    if (ok) ok = EVP_DigestFinal_ex(ctx, root_hash, &hash_length) == 1;
    EVP_MD_CTX_free(ctx);
    REQUIRE(ok && hash_length == 32 && memcmp(root_hash, data+88, 32) == 0);

    const unsigned char *index = sections[2];
    REQUIRE(lengths[2] >= 48 && memcmp(index, "QIDX", 4) == 0);
    REQUIRE(u32(index+4) == 1 && u32(index+8) == 1 && u32(index+12) == 1);
    uint64_t slen = u64(index+16), shlen = u64(index+24), rlen = u64(index+32), clen = u64(index+40);
    const unsigned char *strings = region(index, (size_t)lengths[2], 48, slen);
    REQUIRE(strings != NULL);
    uint64_t shape_start = 48+slen;
    const unsigned char *shapes = region(index, (size_t)lengths[2], shape_start, shlen);
    REQUIRE(shapes != NULL && rlen == 120 && clen == 72);
    uint64_t record_start = shape_start+shlen;
    const unsigned char *record = region(index, (size_t)lengths[2], record_start, rlen);
    REQUIRE(record != NULL);
    const unsigned char *chunk = region(index, (size_t)lengths[2], record_start+rlen, clen);
    REQUIRE(chunk != NULL && record_start+rlen+clen == lengths[2]);
    uint32_t name_ref = u32(record+8), shape_ref = u32(record+12);
    const unsigned char *prefix = region(strings, (size_t)slen, name_ref, 4);
    REQUIRE(prefix != NULL && u32(prefix) == 6);
    const unsigned char *name = region(strings, (size_t)slen, (uint64_t)name_ref+4, 6);
    REQUIRE(name != NULL && memcmp(name, "weight", 6) == 0);
    const unsigned char *shape = region(shapes, (size_t)shlen, shape_ref, 12);
    REQUIRE(shape != NULL && u32(shape) == 1 && u64(shape+4) == 2);
    REQUIRE(u64(record) == 0 && u32(record+16) == 1 && u32(record+20) == 1);
    REQUIRE(u32(record+24) == 0 && u32(record+28) == 1 && u32(record+32) == 0);
    REQUIRE(u32(record+36) == 0 && u32(record+40) == 0 && u32(record+44) == 1);
    REQUIRE(u64(record+48) == 8 && u64(record+56) == 8 && u64(record+64) == 0);
    REQUIRE(u64(record+104) == UINT64_MAX && u32(record+112) == 0 && u32(record+116) == 0);
    REQUIRE(u32(chunk) == 0 && u32(chunk+4) == 0 && u64(chunk+8) == 0);
    REQUIRE(u64(chunk+16) == 8 && u64(chunk+24) == 0 && u64(chunk+32) == 8);
    const unsigned char *payload = region(sections[3], (size_t)lengths[3], u64(chunk+24), u64(chunk+32));
    REQUIRE(payload != NULL && digest(payload, 8, payload_hash));
    REQUIRE(memcmp(payload_hash, record+72, 32) == 0 && memcmp(payload_hash, chunk+40, 32) == 0);
    const unsigned char expected[8] = {0,0,128,63,0,0,0,64};
    REQUIRE(memcmp(payload, expected, 8) == 0);
    if (!json) { puts("qif-c-golden: ok"); return 0; }
    printf("{\"header_hex\":\""); hex(data, 144); printf("\"");
    hash_field("manifest_sha256", hashes[0]); hash_field("metadata_sha256", hashes[1]);
    hash_field("index_sha256", hashes[2]); hash_field("tensor_sha256", record+72); hash_field("chunk_sha256", chunk+40);
    printf(",\"index_counts\":[%u,%u],\"table_lengths\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]", u32(index+8),u32(index+12),slen,shlen,rlen,clen);
    printf(",\"tensor_id\":%" PRIu64 ",\"name\":\"%.*s\",\"shape\":[%" PRIu64 "]",u64(record),6,name,u64(shape+4));
    printf(",\"refs\":[%u,%u],\"encoding\":[%u,%u,%u,%u,%u]",name_ref,shape_ref,u32(record+16),u32(record+20),u32(record+24),u32(record+28),u32(record+32));
    printf(",\"flags\":[%u,%u,%u,%u,%u]",u32(record+36),u32(record+40),u32(record+44),u32(record+112),u32(record+116));
    printf(",\"lengths\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],\"alias_target\":%" PRIu64,u64(record+48),u64(record+56),u64(record+64),u64(record+104));
    printf(",\"chunk\":[%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "]",u32(chunk),u32(chunk+4),u64(chunk+8),u64(chunk+16),u64(chunk+24),u64(chunk+32));
    printf(",\"payload_hex\":\""); hex(payload,8); puts("\"}");
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2],"--json") != 0)) return 2;
    FILE *file = fopen(argv[1],"rb");
    if (!file) return 3;
    if (fseek(file,0,SEEK_END) != 0) { fclose(file); return 3; }
    long length = ftell(file);
    if (length < 144 || length > 1024*1024 || fseek(file,0,SEEK_SET) != 0) { fclose(file); return 4; }
    unsigned char *data = malloc((size_t)length);
    if (!data) { fclose(file); return 3; }
    if (fread(data,1,(size_t)length,file) != (size_t)length) { free(data); fclose(file); return 3; }
    fclose(file);
    int result = inspect(data,(size_t)length,argc == 3);
    free(data);
    return result;
}
