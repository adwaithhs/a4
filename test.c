#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

int main(int argc, char const *argv[])
{
    int n = 100;
    uint32_t *a = malloc(2*n*sizeof(uint32_t));
    for (int i = 0; i < 2*n; i++) {
        a[i] = rand() % 100;
    }
    sizeof(a);
    FILE *fs = fopen("inp", "wb");
    fwrite(a, 8, n, fs);
    fclose(fs);
    return 0;
}
