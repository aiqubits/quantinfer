#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t u64(const unsigned char *p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) value = (value << 8) | p[i];
    return value;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (!file) return 3;
    unsigned char header[144];
    if (fread(header, 1, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "QIF\0", 4) != 0 || u32(header + 4) != 2 ||
        u32(header + 12) != 144 || u32(header + 16) != 1) {
        fclose(file);
        return 4;
    }
    if (fseek(file, (long)u64(header + 56), SEEK_SET) != 0) {
        fclose(file);
        return 5;
    }
    unsigned char index_header[48];
    if (fread(index_header, 1, sizeof(index_header), file) != sizeof(index_header) ||
        memcmp(index_header, "QIDX", 4) != 0 || u32(index_header + 8) != 1 ||
        u32(index_header + 12) != 1) {
        fclose(file);
        return 6;
    }
    if (fseek(file, (long)u64(header + 72), SEEK_SET) != 0) {
        fclose(file);
        return 7;
    }
    unsigned char payload[8];
    if (fread(payload, 1, sizeof(payload), file) != sizeof(payload) ||
        payload[0] != 0 || payload[1] != 0 || payload[2] != 0x80 ||
        payload[3] != 0x3f) {
        fclose(file);
        return 8;
    }
    fclose(file);
    puts("qif2-c-golden: ok");
    return 0;
}
