#include <iostream>
#include <stdio.h>
#include <iostream>
#include <stdio.h>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hiprand.h>
#include <hiprand_kernel.h>
#else
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#endif

#ifdef __HIPCC__
#define CUDA_PRINT(__FORMAT, ...)              \
        printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
        hipThreadIdx_x, hipThreadIdx_y, hipBlockIdx_x, hipBlockIdx_y, ##__VA_ARGS__);
#else
#define CUDA_PRINT(__FORMAT, ...)              \
        printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
#endif

template <typename T>
static void print_tensor_data(const T * data, const size_t numel, const size_t stride, const char * name) {
  printf("=============%s============\n", name);
  size_t index = 0;
  while(index < numel) {
    printf("%2.1f ", data[index]);
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

template <typename T, typename MaskType>
__global__ void RandomGenerator(const size_t n, uint64_t seed,
                                const float dropout_prob, const T* src,
                                MaskType* mask_data, T* dst,
                                bool is_upscale_in_train, uint64_t increment) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  // CUDA_PRINT("idx=%d", idx);
  // CUDA_PRINT("idx=%d gridDim.x=%d gridDim.y=%d gridDim.z=%d blockDim.x=%d blockDim.y=%d blockDim.z=%d", 
  //            idx, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
#ifdef __HIPCC__
  hiprandStatePhilox4_32_10_t state;
  hiprand_init(seed, idx, increment, &state);
#else
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, increment, &state);
#endif

  MaskType mask;
  T dest;
  for (; idx < n; idx += blockDim.x * gridDim.x) {
    T s = src[idx];
    // CUDA_PRINT("idx=%d s=%2.1f", idx, s);
#ifdef __HIPCC__
    if (hiprand_uniform(&state) < dropout_prob) {
#else
    if (curand_uniform(&state) < dropout_prob) {
#endif
      mask = 0;
      dest = 0;
      // CUDA_PRINT("==0== idx=%d s=%2.1f mask=%d dest=%2.1f", idx, s, mask, dest);
    } else {
      mask = 1;
      if (is_upscale_in_train) {
        dest = s / static_cast<T>(1.0f - dropout_prob);
      } else {
        dest = s;
      }
      // CUDA_PRINT("==2== idx=%d s=%2.1f mask=%d dest=%2.1f", idx, s, mask, dest);
    }
    mask_data[idx] = mask;
    dst[idx] = dest;
    // CUDA_PRINT("idx=%d src[idx]=%2.1f dst[idx]=%2.1f mask_data[idx]=%d", 
    //             idx, static_cast<float>(src[idx]), static_cast<float>(dst[idx]), static_cast<int>(mask_data[idx]));
  }
}

int main(void)
{
    // input
    const int input_height = 4;
    const int input_width = 4;
    const int input_numel = input_height * input_width;
    // attr
    const float dropout_prob = 0.5;
    const bool upscale_in_train = false;

    float * input_cpu = new float[input_numel];
    float * output_cpu = new float[input_numel];
    uint8_t * mask_cpu = new uint8_t[input_numel];

    for (int i = 0; i < input_numel; ++i) { input_cpu[i] = 1.0; }

    float * input_gpu, * output_gpu;
    uint8_t * mask_gpu;

#ifdef __HIPCC__
    hipMalloc((void **)&input_gpu, input_numel * sizeof(float));
    hipMalloc((void **)&output_gpu, input_numel * sizeof(float));
    hipMalloc((void **)&mask_gpu, input_numel * sizeof(uint8_t));
    hipMemcpy(input_gpu, input_cpu, input_numel * sizeof(float), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(RandomGenerator<float, uint8_t>),
      dim3(1), dim3(input_numel), 0, 0,
      input_numel, 3, dropout_prob, input_gpu, mask_gpu, output_gpu,
      upscale_in_train, input_numel);

    hipMemcpy(output_cpu, output_gpu, input_numel * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(mask_cpu, mask_gpu, input_numel * sizeof(uint8_t), hipMemcpyDeviceToHost);
#else
    cudaMalloc((void **)&input_gpu, input_numel * sizeof(float));
    cudaMalloc((void **)&output_gpu, input_numel * sizeof(float));
    cudaMalloc((void **)&mask_gpu, input_numel * sizeof(uint8_t));
    cudaMemcpy(input_gpu, input_cpu, input_numel * sizeof(float), cudaMemcpyHostToDevice);

    RandomGenerator<float, uint8_t><<<dim3(1), dim3(input_numel), 0, 0>>>(
      input_numel, 3, dropout_prob, input_gpu, mask_gpu, output_gpu,
      upscale_in_train, input_numel);

    cudaMemcpy(output_cpu, output_gpu, input_numel * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(mask_cpu, mask_gpu, input_numel * sizeof(uint8_t), cudaMemcpyDeviceToHost);
#endif

    print_tensor_data<float>(output_cpu, input_numel, input_width, "output");

    return 0;
}