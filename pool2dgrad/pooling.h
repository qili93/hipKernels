#pragma once
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#define HOSTDEVICE __host__ __device__

// #ifdef __HIPCC__
// #define CUDA_PRINT(__FORMAT, ...)              \
//         printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
//         hipThreadIdx_x, hipThreadIdx_y, hipBlockIdx_x, hipBlockIdx_y, ##__VA_ARGS__);
// #else
// #define CUDA_PRINT(__FORMAT, ...)              \
//         printf("[tid.x=<%d> tid.y=<%d> bid.x=<%d> bid.y=<%d>]: " __FORMAT "\n", \
//         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
// #endif

template <class T>
class MaxPoolGrad {
 public:
  HOSTDEVICE inline void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
    *dx += dy * static_cast<T>(x == y);
  }
};

template <class T>
class AvgPoolGrad {
 public:
  HOSTDEVICE inline void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
    *dx += (scale * dy);
  }
};

HOSTDEVICE inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

HOSTDEVICE inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

template <typename PoolProcess, typename T>
class Pool2dGradFunctor {
 public:
  void operator()(const int batch_size,
                  const int input_channels,
                  const int input_height,
                  const int input_width,
                  const int output_height,
                  const int output_width,
                  const int ksize_height,
                  const int ksize_width,
                  const int stride_height,
                  const int stride_width,
                  const int padding_height,
                  const int padding_width,
                  const T* input_cpu,
                  const T* output_cpu,
                  const T* output_grad_cpu,
                  T* input_grad_cpu,
                  PoolProcess pool_process,
                  const bool exclusive, 
                  const bool adaptive);
};

