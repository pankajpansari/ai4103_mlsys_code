#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

__global__ void matMulKernel(float *x, float *y, float *z, int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < N)
    {
        z[i * N + j] = 0.0f;
        for (int m = 0; m < N; m++)
        {
            z[i * N + j] += (x[i * N + m] * y[m * N + j]);
        }
    }
}

void multiply_mat(float *x, float *y, float *z, int N)
{
    float *x_d, *y_d, *z_d;
    int size = N * N * sizeof(float);

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);
    cudaMalloc((void **)&z_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    int a = (int)MAX(N / 16, 1);
    int b = (int)MAX(N / 16, 1);

    dim3 dimGrid(N / 16, N / 16, 1);
    dim3 dimBlock(16, 16, 1);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matMulKernel<<<dimGrid, dimBlock>>>(x_d, y_d, z_d, N);

    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main()
{
    int N = 32; // Square matrices of shape N x N (here N = 32)
    size_t size = N * N * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);
    float *z = (float *)malloc(size);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            x[i * N + j] = 2.0f;
            y[i * N + j] = 1.0f;
            z[i * N + j] = 0.0f;
        }
    }

    multiply_mat(x, y, z, N);

    float max_err = 0.0f;

    // Check if all values in result are 2 * N
    float target = 2 * N;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            max_err = std::fmax(max_err, std::fabs(target - z[i * N + j]));
        }
    }

    std::cout << "max error = " << max_err << std::endl;

    free(x);
    free(y);
    free(z);

    return 0;
}