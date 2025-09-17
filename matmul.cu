#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matMulKernel(const float *x, const float *y, float *out, const int N)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    if (i < N && j < N)
    {
        float out_val = 0.0f;
        for (int m = 0; m < N; m++)
        {
            out_val += (x[i * N + m] * y[m * N + j]);
        }
        out[i * N + j] = out_val;
    }
}

void multiply_mat(const float *x, const float *y, float *out, const int N)
{
    float *x_d, *y_d, *out_d;
    int size = N * N * sizeof(float);

    cudaMalloc((void **)&x_d, size);
    cudaMalloc((void **)&y_d, size);
    cudaMalloc((void **)&out_d, size);

    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    dim3 dimGrid(grid_size, grid_size, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    matMulKernel<<<dimGrid, dimBlock>>>(x_d, y_d, out_d, N);

    cudaMemcpy(out, out_d, size, cudaMemcpyDeviceToHost);

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(out_d);
}

int main()
{
    int N = 32; // Square matrices of shape N x N (here N = 32)
    size_t size = N * N * sizeof(float);

    float *x = (float *)malloc(size);
    float *y = (float *)malloc(size);
    float *out = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        x[i] = 2.0f;
        y[i] = 1.0f;
    }

    multiply_mat(x, y, out, N);

    float max_err = 0.0f;

    // Check if all values in result are 2 * N
    float target = 2 * N;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            max_err = std::fmax(max_err, std::fabs(target - out[i * N + j]));

    std::cout << "max error = " << max_err << std::endl;

    free(x);
    free(y);
    free(out);

    return 0;
}