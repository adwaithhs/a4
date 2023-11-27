#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <getopt.h>
#include <sys/stat.h>
#include <sys/types.h>

#define KEY(x) *((uint32_t*)x)

int comp32(const void *elem1, const void *elem2) 
{
    uint32_t f = KEY(elem1);
    uint32_t s = KEY(elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

void print_array(uint64_t *arr, uint64_t n) {
    for (int i = 0; i < n; i++) {
        printf("%" PRIu32 " ", KEY(arr[i]));
    }
    printf("\n");
}

uint64_t* cpu_sort(uint64_t* data, uint64_t n, int p) {
    if (n <= 4*p*p)
    {
        qsort(data, n, 8, comp32);
        return data;
    }
    uint64_t q = n / p;
    int r = n % p;
    
    uint64_t *R = (uint64_t*)malloc(p*p * sizeof(*R));
    uint64_t *S = (uint64_t*)malloc((p-1) * sizeof(*S));
    uint64_t *m = (uint64_t*)malloc(p * sizeof(*m));
    uint64_t *c = (uint64_t*)malloc(p * sizeof(*c));
    uint64_t *h = (uint64_t*)malloc(p * sizeof(*h));
    uint64_t *final = (uint64_t*)malloc(n * sizeof(*final));
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            uint64_t a = 0;
            for (int i = 0; i < p; i++)
            {
                uint64_t size;
                if (i < r)
                {
                    size = q+1;
                }
                else
                {
                    size = q;
                }
                #pragma omp task
                qsort(data + a, size, 8, comp32);
                a += size;
            }
            #pragma omp taskwait
            int k = 0;
            a = 0;
            for (int i = 0; i < p; i++)
            {
                uint64_t size;
                if (i < r)
                {
                    size = q+1;
                }
                else
                {
                    size = q;
                }
                

                for (uint64_t j = a; j < a + size; j+= size/p)
                {
                    R[k] = data[j];
                    k++;
                }
                
                a += size;
            }

            qsort(R, p*p, 8, comp32);
            for (int j = 0; j < p-1; j++)
            {
                S[j] = R[(j+1)*p];
            }
            for (int j = 0; j < p; j++)
            {
                #pragma omp task
                {
                    m[j] = 0;
                    for (uint64_t i = 0; i < n; i++)
                    {
                        if ((j == 0 || S[j-1] < data[i]) && (j == p-1 || data[i] <= S[j]))
                        {
                            m[j]++;
                        }
                    }
                }
            }
            #pragma omp taskwait
            c[0] = 0;
            h[0] = 0;
            for (int i = 1; i < p; i++)
            {
                c[i] = c[i-1] + m[i-1];
                h[i] = c[i];
            }

            for (int j = 0; j < p; j++)
            {
                #pragma omp task
                {
                    uint64_t k = 0;
                    for (uint64_t i = 0; i < n; i++)
                    {
                        uint32_t key = KEY(data[i]);
                        if ((j == 0 || S[j-1] < key) && (j == p-1 || key <= S[j]))
                        {
                            final[c[j]+k] = data[i];
                        }
                    }
                }
            }
            #pragma omp taskwait

            for (int j = 0; j < p; j++)
            {
                #pragma omp task 
                qsort(final + c[j], m[j], 8, comp32);
            }
            #pragma omp taskwait
        }
    }
    return final;
}

__global__ void step(uint64_t *ddata, int j, int k)
{
    int i, l;
    i = threadIdx.x + blockDim.x * blockIdx.x;
    l = i^j;

    if (l > i) {
        if ((i&k)==0) {
            if (KEY(ddata[i]) > KEY(ddata[l])) {
                uint64_t temp = ddata[i];
                ddata[i] = ddata[l];
                ddata[l] = temp;
            }
        }
        if ((i&k)!=0) {
            if (KEY(ddata[i]) < KEY(ddata[l])) {
                uint64_t temp = ddata[i];
                ddata[i] = ddata[l];
                ddata[l] = temp;
            }
        }
    }
}


void gpu_sort(uint64_t* data, uint64_t n, uint64_t blocks, uint64_t threads) {
    uint64_t *ddata;

    cudaMalloc(&ddata, n*sizeof(uint64_t));
    cudaMemcpy(ddata, data, n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    int j, k;
    for (k = 2; k <= n; k <<= 1) {
        for (j=k>>1; j>0; j=j>>1) {
            step<<<blocks, threads>>>(ddata, j, k);
        }
    }
    cudaMemcpy(data, ddata, n*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ddata);
}

uint64_t * merge(uint64_t* a, uint64_t n, uint64_t* b, uint64_t m) {
    uint64_t *c = (uint64_t*)malloc((n*m) * sizeof(*c));
    uint64_t i = 0, j = 0, k = 0;
    
    while (i < n && j < m) {
        if (a[i] < b[j]) {
            c[k] = a[i];
            i++;
        } else {
            c[k] = b[j];
            j++;
        }
        k++;
    }

    while (i < n) {
        c[k] = a[i];
        i++;
        k++;
    }

    while (j < m) {
        c[k] = b[j];
        j++;
        k++;
    }
    return c;
}

int main(int argc, char** argv) {
    static struct option long_options[] = {
        {"inputname", required_argument, 0, 'i'},
        {"outputpath", required_argument, 0, 'o'},
        {"p", required_argument, 0, 'p'},
        {"gpu", no_argument, 0, 'g'},
        {0, 0, 0, 0}
    };
    int p, use_gpu = 0;
    char _i[] = "input";
    char _o[] = "output";
    char *input_path = _i;
    char *output_path = _o;
    while (1) {
        int c = getopt_long(argc, argv, "", long_options, NULL);
        if (c == -1) break;
        switch (c)
        {
        case 'i':
            input_path = (char*)malloc((strlen(optarg)+1)*sizeof(char));
            strcpy(input_path, optarg);
            break;
        case 'o':
            output_path = (char*)malloc((strlen(optarg)+1)*sizeof(char));
            strcpy(output_path, optarg);
            break;
        case 'p':
            p = atoi(optarg);
            break;
        case 'g':
            use_gpu = 1;
            break;
        
        default:
            break;
        }
    }

    
    uint64_t* data;

    FILE *fs = fopen(input_path, "rb");
    fseeko(fs, 0, SEEK_END);
    off_t n = ftello(fs) / 8;


    fseeko(fs, 0, SEEK_SET);
    data = (uint64_t*)malloc(n*8);
    fread(data, 8, n, fs);
    fclose(fs);
    
    int nDevices = 0;
    cudaGetDeviceCount(&nDevices);

    if (n <= 4*p*p)
    {
        qsort(data, n, 8, comp32);
        FILE *fs = fopen(output_path, "wb");
        fwrite(data, 8, n, fs);
        fclose(fs);
        return;
    }

    if (use_gpu == 0 || nDevices == 0) {
        uint64_t* final = cpu_sort(data, n, p);
        FILE *fs = fopen(output_path, "wb");
        fwrite(final, 8, n, fs);
        fclose(fs);
        return;
    }
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int threads = prop.maxThreadsPerBlock;

    uint64_t r = 1;
    while (2*r < n) {
        r = 2*r;
    }
    if (r==n) {
        if (r <= threads) {
            gpu_sort(data, n, 1, n);
        } else {
            gpu_sort(data, n, n/threads, threads);
        }
        FILE *fs = fopen(output_path, "wb");
        fwrite(data, 8, n, fs);
        fclose(fs);
        return;
    }
    int b, t;
    if (r <= threads) {
        b = 1;
        t = r;
    } else {
        b = r/threads;
        t = threads;
    }

    uint64_t *ddata;

    cudaMalloc(&ddata, n*sizeof(uint64_t));
    cudaMemcpy(ddata, data, n*sizeof(uint64_t), cudaMemcpyHostToDevice);

    int j, k;
    for (k = 2; k <= n; k <<= 1) {
        for (j=k>>1; j>0; j=j>>1) {
            step<<<b, t>>>(ddata, j, k);
        }
    }
    uint64_t* final = cpu_sort(data + r, n - r, p);
    cudaMemcpy(data, ddata, n*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(ddata);

    final = merge(data, r, final, n-r);
    fs = fopen(output_path, "wb");
    fwrite(final, 8, n, fs);
    fclose(fs);

    return 0;
}
