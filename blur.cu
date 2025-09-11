#include "blur.cuh"
#include "ErrorCheck.cuh"
#include <cstdio>
#include <cuda_runtime.h>

__device__
void cuda_blur_kernel_convolution(uint raw_data_index, const float* gpu_raw_data,
                                  const float* gpu_blur_v, float* gpu_out_data,
                                  const unsigned int n_frames,
                                  const unsigned int blur_v_size) {
  if (raw_data_index < blur_v_size) {
    for (int j = 0; j <= raw_data_index; j++)
      gpu_out_data[raw_data_index] += gpu_raw_data[raw_data_index - j] * gpu_blur_v[j]; 
  } else if (raw_data_index >= blur_v_size && raw_data_index < n_frames) {
    for (int j = 0; j < blur_v_size; j++)
      gpu_out_data[raw_data_index] += gpu_raw_data[raw_data_index - j] * gpu_blur_v[j]; 
    }

}

__global__
void cuda_blur_kernel(const float *gpu_raw_data, const float *gpu_blur_v,
                      float *gpu_out_data, int n_frames, int blur_v_size) {
    uint raw_data_index = blockIdx.x * blockDim.x + threadIdx.x;

    while (raw_data_index < n_frames) {
        cuda_blur_kernel_convolution(raw_data_index, gpu_raw_data,
                                     gpu_blur_v, gpu_out_data,
                                     n_frames, blur_v_size);
        raw_data_index += gridDim.x * blockDim.x;
    }
}


void cuda_call_blur_kernel(const unsigned int blocks,
                            const unsigned int threads_per_block,
                            const float *raw_data,
                            const float *blur_v,
                            float *out_data,
                            const unsigned int n_frames,
                            const unsigned int blur_v_size) {
    float* gpu_raw_data;
    cudaMalloc(&gpu_raw_data, n_frames * sizeof(float));

    cudaMemcpy(gpu_raw_data, raw_data, n_frames * sizeof(float), cudaMemcpyHostToDevice);
    float* gpu_blur_v;
    cudaMalloc(&gpu_blur_v, blur_v_size * sizeof(float));

    cudaMemcpy(gpu_blur_v, blur_v, blur_v_size * sizeof(float), cudaMemcpyHostToDevice);

    float* gpu_out_data;
    cudaMalloc(&gpu_out_data, n_frames * sizeof(float));
    cudaMemset(gpu_out_data, 0, n_frames * sizeof(float));
    
    cuda_blur_kernel<<<(int) blocks, (int) threads_per_block>>>(gpu_raw_data, gpu_blur_v, gpu_out_data, n_frames, blur_v_size);


    cudaMemcpy(out_data, gpu_out_data, n_frames * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpu_raw_data);
    cudaFree(gpu_blur_v);
    cudaFree(gpu_out_data); 

}
