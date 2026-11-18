/* Standalone test CLI: no QuantInfer, Rust or Python binding is linked. */
#include "qif.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
  if (setjmp(qif_error)) {
    cleanup();
    return 1;
  }
  CHECK(argc >= 2 && argc % 2 == 0);
  Json trusted = array();
  for (int i = 2; i < argc; i += 2) {
    CHECK(strcmp(argv[i], "--trusted-key") == 0 && strlen(argv[i + 1]) == 64);
    char key[65];
    for (size_t j = 0; j < 64; j++) {
      unsigned char c = (unsigned char)argv[i + 1][j];
      if (c >= 'A' && c <= 'F')
        c += 32;
      CHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
      key[j] = (char)c;
    }
    key[64] = 0;
    push(trusted, string(key));
  }
  Json result = report(argv[1], trusted);
  Bytes data = serialized(result);
  CHECK(fwrite(data.data, 1, data.size, stdout) == data.size &&
        putchar('\n') != EOF);
  cleanup();
  return 0;
}
