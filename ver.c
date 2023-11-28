#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>

#define KEY(x) *((uint32_t*)&x)

int main(int argc, char const *argv[])
{
    uint64_t* data;

    FILE *fs = fopen("out", "rb");
    fseeko(fs, 0, SEEK_END);
    off_t n = ftello(fs) / 8;

    printf("%" PRIu64 "\n", n);

    fseeko(fs, 0, SEEK_SET);
    data = (uint64_t*)malloc(n*8);
    fread(data, 8, n, fs);
    fclose(fs);
    for (uint64_t i = 1; i < n; i++) {
        if (KEY(data[i]) < KEY(data[i-1])) {
            printf("NOT OK %" PRIu64 "\n", i);
        }
    }
    printf("OK\n");
    return 0;
}
