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
    int size = N * sizeof(float);
    cudaMemPrefetchAsync(x, size, 0, 0);
    cudaMemPrefetchAsync(y, size, 0, 0);

    int blockSize = 256;
    int numBlocks = ceil(N / 256.0);
    vecAddKernel<<<numBlocks, blockSize>>>(x_d, y_d, N);
    cudaDeviceSynchronize();
}

int main()
{
    int N = 1 << 20; // About 1M elements
    size_t size = N * sizeof(float);

    float *x, *y;
    cudaMallocManaged(&x, size);
    cudaMallocManaged(&y, size);

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

    cudaFree(x);
    cudaFree(y);

    return 0;
}