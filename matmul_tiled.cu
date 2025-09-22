#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matMulKernel(const float *a, const float *b, float *out, const int N)
{
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float a_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float b_tile[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float out_val = 0.0f;

    for (int k = 0; k < N / BLOCK_SIZE; k++)
    {
        int offset = k * BLOCK_SIZE;
        a_tile[ty][tx] = a[row * N + offset + tx];
        b_tile[ty][tx] = b[(offset + ty) * N + col];
        __syncthreads();

        for (int m = 0; m < BLOCK_SIZE; m++)
            out_val += (a_tile[ty][m] * b_tile[m][tx]);
        __syncthreads();
    }

    out[row * N + col] = out_val;
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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <mat_size>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]); // Square matrices of shape N x N (N >= 32 ideally)
 
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