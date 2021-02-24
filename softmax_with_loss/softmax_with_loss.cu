#include <iostream>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
        hipThreadIdx_x, hipThreadIdx_y, hipBlockIdx_x, hipBlockIdx_y, ##__VA_ARGS__);
#else
#include <cuda.h>
#include "cub/cub.cuh"
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
#endif

template <typename T>
static void print_data_2d(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride = dims[1];
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", data[index]);
    } else {
      printf("%d ", data[index]);
    }
    if((index+1) % stride == 0) printf("\n");
    index ++;
  }
}

template <typename T>
struct TolerableValue {
  __host__ __device__ T operator()(const T& x) const {
    const T kApproInf = 1e20;

    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

static __device__ __forceinline__ float exp_on_device(float x) {
  return expf(x);
}
static __device__ __forceinline__ double exp_on_device(double x) {
  return exp(x);
}
static __device__ __forceinline__ float log_on_device(float x) {
  return TolerableValue<float>()(logf(x));
}
static __device__ __forceinline__ double log_on_device(double x) {
  return TolerableValue<double>()(log(x));
}


template <typename T, int BlockDim>
#ifdef __HIPCC__
using BlockReduce = hipcub::BlockReduce<T, BlockDim /*, hipcub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;
#else
using BlockReduce = cub::BlockReduce<T, BlockDim /*, cub::BLOCK_REDUCE_WARP_REDUCTIONS*/>;
#endif

template <typename T, int BlockDim>
using BlockReduceTempStorage = typename BlockReduce<T, BlockDim>::TempStorage;

template <typename T, int BlockDim>
static __global__ void RowReductionForMax(const T* logits_data, T* max_data,
                                          int d, int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits_data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  KERNEL_PRINT("remain=%d idx_n=%d idx_remain=%d beg_idx=%d end_idx=%d", 
                remain, idx_n, idx_remain, beg_idx, end_idx);

  int step = BlockDim * remain;
  T cur_max = logits_data[beg_idx];
  KERNEL_PRINT("beg_idx=%d logits_data[beg_idx]=%f cur_max=%f", 
                beg_idx, static_cast<float>(logits_data[beg_idx]), static_cast<float>(cur_max));
  beg_idx += step;
  while (beg_idx < end_idx) {
    if (cur_max < logits_data[beg_idx]) {
      cur_max = logits_data[beg_idx];
    }
    beg_idx += step;
  }
#ifdef __HIPCC__
  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, hipcub::Max());
#else
  cur_max = BlockReduce<T, BlockDim>(temp_storage).Reduce(cur_max, cub::Max());
#endif

  if (threadIdx.x == 0) max_data[blockIdx.x] = cur_max;
  KERNEL_PRINT("max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
}

// Make sure that BlockDim <= axis_dim
template <typename T, int BlockDim, bool CalculateLogSoftmax = false>
static __global__ void RowReductionForDiffMaxSum(const T* logits_data,
                                                 T* max_data, T* softmax, int d,
                                                 int axis_dim) {
  __shared__ BlockReduceTempStorage<T, BlockDim> temp_storage;

  // logits, softmax data view as [n, axis_dim, remain]
  // max_data view as [n, 1, remain]
  // blockDim = n * remain, split blockIdx to idx_n and idx_remain
  int remain = d / axis_dim;
  int idx_n = blockIdx.x / remain;
  int idx_remain = blockIdx.x % remain;
  int beg_idx = idx_n * d + threadIdx.x * remain + idx_remain;
  int end_idx = (idx_n + 1) * d;

  auto block_max = max_data[blockIdx.x];
  int step = BlockDim * remain;

  KERNEL_PRINT("remain=%d idx_n=%d idx_remain=%d beg_idx=%d end_idx=%d step=%d", 
                remain, idx_n, idx_remain, beg_idx, end_idx, step);

  // In numeric stable mode softmax_with_loss, we calc loss with
  // tmp_i_j = x_i_j - max_i - logDiffMaxSum_i, instead of
  // log(exp(x_i_j - max_i)/DiffMaxSum_i). Therefore, log(0) will not occur.
  // Also we calc softmax_i_j = e^{tmp_i_j}, the maximum and minimum value will
  // be 1.0 and 0.0, represent prob is 1.0 and 0.0.
  // So there is no need to clip on shift_softmax.
  softmax[beg_idx] = logits_data[beg_idx] - block_max;
  T diff_max_sum = exp_on_device(softmax[beg_idx]);
  KERNEL_PRINT("beg_idx=%d logits_data[beg_idx]=%f block_max=%f softmax[beg_idx]=%f diff_max_sum=%f", 
                beg_idx, static_cast<float>(logits_data[beg_idx]),
                static_cast<float>(block_max),
                static_cast<float>(softmax[beg_idx]),
                static_cast<float>(diff_max_sum));
  auto idx = beg_idx + step;
  while (idx < end_idx) {
    softmax[idx] = logits_data[idx] - block_max;
    diff_max_sum += exp_on_device(softmax[idx]);
    KERNEL_PRINT("idx=%d logits_data[idx]=%f block_max=%f softmax[idx]=%f diff_max_sum=%f", 
                  idx, static_cast<float>(logits_data[idx]),
                  static_cast<float>(block_max),
                  static_cast<float>(softmax[idx]),
                  static_cast<float>(diff_max_sum));
    idx += step;
  }
  KERNEL_PRINT("==0== diff_max_sum=%f", static_cast<float>(diff_max_sum));
#ifdef __HIPCC__
  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, hipcub::Sum());
#else
  diff_max_sum =
      BlockReduce<T, BlockDim>(temp_storage).Reduce(diff_max_sum, cub::Sum());
#endif
  
  if (threadIdx.x == 0) {
    max_data[blockIdx.x] = log_on_device(diff_max_sum);
    KERNEL_PRINT("==1== diff_max_sum=%f max_data[blockIdx.x]=%f", static_cast<float>(diff_max_sum), static_cast<float>(max_data[blockIdx.x]));
  }

  if (!CalculateLogSoftmax) return;
  __syncthreads();
  diff_max_sum = max_data[blockIdx.x];
  softmax[beg_idx] -= diff_max_sum;
  KERNEL_PRINT("max_data[blockIdx.x]=%f diff_max_sum=%f beg_idx=%d softmax[beg_idx]=%f",
                static_cast<float>(max_data[blockIdx.x]),
                static_cast<float>(diff_max_sum),
                beg_idx,
                static_cast<float>(softmax[beg_idx]));

  beg_idx += step;
  while (beg_idx < end_idx) {
    softmax[beg_idx] -= diff_max_sum;
    KERNEL_PRINT("beg_idx=%d end_idx=%d diff_max_sum=%f softmax[beg_idx]=%f", 
                  beg_idx, end_idx, 
                  static_cast<float>(diff_max_sum),
                  static_cast<float>(softmax[beg_idx]));
    beg_idx += step;
  }

  // Note(zhiqiu): since different threads may use max_data[blockIdx.x] to
  // calculate diff_max_sum, __syncthreads() is needed here.
  __syncthreads();
  if (threadIdx.x == 0) max_data[blockIdx.x] = 0;
  // KERNEL_PRINT("max_data[blockIdx.x]=%f", static_cast<float>(max_data[blockIdx.x]));
}

int main(void)
{
  // dims and numel
  const std::vector<int64_t> logit_vec = {2, 3};
  const int64_t logit_numel = std::accumulate(logit_vec.begin(), logit_vec.end(), 1, std::multiplies<int64_t>());
  const std::vector<int64_t> label_vec = {2, 1};
  const int64_t label_numel = std::accumulate(label_vec.begin(), label_vec.end(), 1, std::multiplies<int64_t>());
  // attrs
  const int n = 2;
  const int d = 3;
  const int axis_dim = 3;
  const int ignore_idx = -1;
  // 
  const int block_dim = 2; // 2
  const int grid_dim = n * d / axis_dim; // 2

  // input
  float * logit_cpu = new float[logit_numel];
  float * label_cpu = new float[label_numel];
  // output
  float * softmax_cpu = new float[logit_numel];
  float * loss_cpu = new float[label_numel];

  for (int i = 0; i < logit_numel; ++i) { 
    logit_cpu[i] = i;
  }
  for (int i = 0; i < label_numel; ++i) {
    label_cpu[i] = i;
  }
  print_data_2d<float>(logit_cpu, logit_numel, logit_vec, "logit");
  print_data_2d<float>(label_cpu, label_numel, label_vec, "label");

  float * logit_gpu, * label_gpu;
  float * softmax_gpu, * loss_gpu;

#ifdef __HIPCC__
  hipMalloc((void **)&logit_gpu, logit_numel * sizeof(float));
  hipMalloc((void **)&label_gpu, label_numel * sizeof(float));
  hipMalloc((void **)&softmax_gpu, logit_numel * sizeof(float));
  hipMalloc((void **)&loss_gpu, label_numel * sizeof(float));
  hipMemcpy(logit_gpu, logit_cpu, logit_numel * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(label_gpu, label_cpu, label_numel * sizeof(float), hipMemcpyHostToDevice);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForMax<float, block_dim>),
        dim3(grid_dim), dim3(block_dim), 0, 0,
        logit_gpu, loss_gpu, d, axis_dim);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(RowReductionForDiffMaxSum<float,  block_dim, true>),
        dim3(grid_dim), dim3(block_dim), 0, 0,
        logit_gpu, loss_gpu, softmax_gpu, d, axis_dim);

  hipMemcpy(softmax_cpu, softmax_gpu, logit_numel * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(loss_cpu, loss_gpu, label_numel * sizeof(uint8_t), hipMemcpyDeviceToHost);
#else
  cudaMalloc((void **)&logit_gpu, logit_numel * sizeof(float));
  cudaMalloc((void **)&label_gpu, label_numel * sizeof(float));
  cudaMalloc((void **)&softmax_gpu, logit_numel * sizeof(float));
  cudaMalloc((void **)&loss_gpu, label_numel * sizeof(float));
  cudaMemcpy(logit_gpu, logit_cpu, logit_numel * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(label_gpu, label_cpu, label_numel * sizeof(float), cudaMemcpyHostToDevice);

  RowReductionForMax<float, block_dim><<<dim3(grid_dim), dim3(block_dim), 0, 0>>>(
        logit_gpu, loss_gpu, d, axis_dim);
  RowReductionForDiffMaxSum<float,  block_dim, true><<<dim3(grid_dim), dim3(block_dim), 0, 0>>>(
        logit_gpu, loss_gpu, softmax_gpu, d, axis_dim);

  cudaMemcpy(softmax_cpu, softmax_gpu, logit_numel * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(loss_cpu, loss_gpu, label_numel * sizeof(uint8_t), cudaMemcpyDeviceToHost);
#endif

  print_data_2d<float>(softmax_cpu, logit_numel, logit_vec, "softmax");
  print_data_2d<float>(loss_cpu, label_numel, label_vec, "loss");

  return 0;
}