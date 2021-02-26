#include "kernel_printf.h"

#define PADDLE_CUDA_NUM_THREADS 4

inline int GET_BLOCKS(const int N) {
  return (N + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS;
}

template <typename T, int BLOCK_W, int BLOCK_H>
__global__ void MatrixColReduce(const T *__restrict__ in, T *__restrict__ out,
                                size_t width, size_t height) {
  __shared__ T sdata[BLOCK_H][BLOCK_W + 1];
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  size_t width_stride = gridDim.x * blockDim.x;
  size_t full_width = (width & (~((uint64_t)(BLOCK_W - 1)))) +
                      ((width & (BLOCK_W - 1)) ? BLOCK_W : 0);
  size_t full_height = (height & (~((uint64_t)(BLOCK_H - 1)))) +
                       ((height & (BLOCK_H - 1)) ? BLOCK_H : 0);

#pragma unroll
  for (size_t w = idx; w < full_width; w += width_stride) {
    sdata[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();
    size_t offset = w + threadIdx.y * width;
#pragma unroll
    for (size_t h = threadIdx.y; h < full_height;
         h += BLOCK_H) {  // block-stride loop across matrix height
      sdata[threadIdx.y][threadIdx.x] +=
          (w < width && h < height) ? in[offset] : (static_cast<T>(0));
      offset += width * BLOCK_H;
    }
    __syncthreads();

    T val = sdata[threadIdx.x][threadIdx.y];
#ifdef __HIPCC__
    for (int i = warpSize >> 1; i > 0; i >>= 1)
      val += __shfl_xor(val, i);
#else
    for (int i = warpSize >> 1; i > 0; i >>= 1)
      val += __shfl_xor_sync(0xFFFFFFFF, val, i);
#endif

    __syncthreads();
    if (threadIdx.x == 0) sdata[0][threadIdx.y] = val;
    __syncthreads();
    if ((threadIdx.y == 0) && ((w) < width)) out[w] = sdata[0][threadIdx.x];
  }
}

int main() {
    const size_t width = 8;
    const size_t height = 2;
    const std::vector<int64_t> input_vec = {8, 2};
    const std::vector<int64_t> output_vec = {8};
    const size_t numel = width * height;
    int * cpu_indata = new int[numel];
    int * cpu_outdata = new int[width];
    for (unsigned i = 0; i < numel; i++) {
        cpu_indata[i] = i;
    }
    for (unsigned i = 0; i < width; i++) {
        cpu_outdata[i] = 0;
    }

    print_data<int>(cpu_indata, numel, input_vec, "input");
    print_data<int>(cpu_outdata, width, output_vec, "output");

    constexpr int block_x = 32;
    constexpr int block_y = 32;
    dim3 blocks(block_x, block_y);

    int max_physical_threads = 1024;
    int max_blocks = std::max(max_physical_threads / (block_x * block_y), 1);
    int theory_block = (width + blocks.x - 1) / blocks.x;
    dim3 grids(std::min(theory_block, max_blocks));

    // const int block_num = GET_BLOCKS(height);
    // printf("block_num=%d\n", block_num);

    // dim3 grids(1);
    // dim3 blocks(block_num);

    int *gpu_indata, *gpu_outdata;
#ifdef __HIPCC__
    hipMalloc((void**)&gpu_indata, numel * sizeof(int));
    hipMalloc((void**)&gpu_outdata, width * sizeof(int));
    hipMemcpy(gpu_indata, cpu_indata, numel * sizeof(int), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(MatrixColReduce<int, block_x, block_y>), 
        dim3(blocks), dim3(grids), 0, 0,
        gpu_indata, gpu_outdata, width, height);
    hipDeviceSynchronize();

    hipMemcpy(cpu_outdata, gpu_outdata, width * sizeof(int), hipMemcpyDeviceToHost);
#else
    cudaMalloc((void**)&gpu_indata, numel * sizeof(int));
    cudaMalloc((void**)&gpu_outdata, width * sizeof(int));
    cudaMemcpy(gpu_indata, cpu_indata, numel * sizeof(int), cudaMemcpyHostToDevice);

    MatrixColReduce<int, block_x, block_y><<<blocks, grids, 0, 0>>>(gpu_indata, gpu_outdata, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(cpu_outdata, gpu_outdata, width * sizeof(int), cudaMemcpyDeviceToHost);
#endif

    print_data<int>(cpu_outdata, width, output_vec, "output");

    return 0;
}
