#include <iostream>
#include <cmath>

__global__ void vecAddKernel(float *x, float *y, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N)
    {
        y[i] = x[i] + y[i];
    }
}

void add_arr(float *x, float *y, int N)
{
    float *x_d, *y_d;
    int size = N * sizeof(float);

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = ceil(N / 256.0);
    vecAddKernel<<<numBlocks, blockSize>>>(x_d, y_d, N);

    cudaMemcpy(y, y_d, size, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
}

int main()
{
    int N = 1 << 20; // About 1M elements
    size_t size = N * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add_arr(x, y, N);

    float max_err = 0.0f;

    // Check if all values in sum are 3
    for (int i = 0; i < N; i++)
    {
        max_err = std::fmax(max_err, std::fabs(3.0 - y[i]));
    }

    std::cout << "max error = " << max_err << std::endl;

    free(x);
    free(y);

    return 0;
}