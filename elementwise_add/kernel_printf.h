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
#else
#include <cuda.h>
#endif

#ifdef __HIPCC__
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("%03d: [tid.x=<%lu> tid.y=<%lu> bid.x=<%lu> bid.y=<%lu>]: " __FORMAT "\n", \
        __LINE__, hipThreadIdx_x, hipThreadIdx_y, hipBlockIdx_x, hipBlockIdx_y, ##__VA_ARGS__);
#else
#define KERNEL_PRINT(__FORMAT, ...)              \
        printf("%03d: [tid.x=<%lu> tid.y=<%lu> bid.x=<%lu> bid.y=<%lu>]: " __FORMAT "\n", \
        __LINE__, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ##__VA_ARGS__);
#endif

template <typename T>
static void print_data_1d(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", data[index]);
    } else {
      printf("%d ", data[index]);
    }
    index ++;
  }
  printf("\n");
}

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
static void print_data_4d(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  printf("------------%s------------\n", name.c_str());
  size_t stride_h = dims[3];
  size_t stride_w = dims[2] * stride_h;
  size_t index = 0;
  while(index < numel) {
    if (std::is_floating_point<T>::value) {
      printf("%f ", data[index]);
    } else {
      printf("%d ", data[index]);
    }
    if((index+1) % stride_h == 0) printf("\n");
    if((index+1) % stride_w == 0) printf("\n");
    index ++;
  }
}

template <typename T>
static void print_data(const T * data, const int64_t numel, const std::vector<int64_t> dims, const std::string name) {
  if(dims.size() == 4UL) {
    return print_data_4d<T>(data, numel, dims, name);
  }
  if(dims.size() == 2UL) {
    return print_data_2d<T>(data, numel, dims, name);
  }
  if(dims.size() == 1UL) {
    return print_data_1d<T>(data, numel, dims, name);
  }
}