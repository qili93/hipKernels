#include "pooling.h"

template <typename PoolProcess, typename T>
__global__ void KernelPool2DGrad(
    const int nthreads, const T* input_gpu, const T* output_gpu,
    const T* output_grad_cpu, const int channels, const int input_height,
    const int input_width, const int output_height, const int output_width,
    const int ksize_height, const int ksize_width, const int stride_height,
    const int stride_width, const int padding_height, const int padding_width,
    PoolProcess pool_process, bool exclusive, bool adaptive, T* input_grad_cpu,
    bool channel_last = false) {
#ifdef __HIPCC__
  for (int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; index < nthreads;
       index += hipBlockDim_x * hipGridDim_x) {
#else
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
#endif
    // CUDA_PRINT("index=%d channel_last=%d channels=%d input_width=%d input_height=%d padding_width=%d padding_height=%d", 
    //             index, channel_last, channels, input_width, input_height, padding_width, padding_height);
    int w_offset, h_offset, offsetC, batch_idx;
    if (!channel_last) { /* NCHW */
      w_offset = index % input_width + padding_width;
      h_offset = (index / input_width) % input_height + padding_height;
      offsetC = (index / input_width / input_height) % channels;
      batch_idx = index / input_width / input_height / channels;
    } else { /* NHWC */
      offsetC = index % channels;
      w_offset = (index / channels) % input_width + padding_width;
      h_offset =
          (index / channels / input_width) % input_height + padding_height;
      batch_idx = index / channels / input_width / input_height;
    }
    // CUDA_PRINT("index=%d w_offset=%d h_offset=%d offsetC=%d batch_idx=%d", 
    //             index, w_offset, h_offset, offsetC, batch_idx);

    int phstart, phend;
    int pwstart, pwend;
    if (adaptive) {
      phstart = AdaptStartIndex(h_offset, output_height, input_height);
      phend = AdaptEndIndex(h_offset, output_height, input_height);

      pwstart = AdaptStartIndex(w_offset, output_width, input_width);
      pwend = AdaptEndIndex(w_offset, output_width, input_width);
    } else {
      phstart = (h_offset < ksize_height)
                    ? 0
                    : (h_offset - ksize_height) / stride_height + 1;
      pwstart = (w_offset < ksize_width)
                    ? 0
                    : (w_offset - ksize_width) / stride_width + 1;
      phend = min(h_offset / stride_height + 1, output_height);
      pwend = min(w_offset / stride_width + 1, output_width);
    }
    T gradient = static_cast<T>(0.0);
    T input_cpu = input_gpu[index];
    // CUDA_PRINT("index=%d adaptive=%d phstart=%d phend=%d pwstart=%d pwend=%d", 
    //             index, adaptive, phstart, phend, pwstart, pwend);
    // CUDA_PRINT("index=%d gradient=%1.1f input_cpu=%1.1f", 
    //             index, gradient, input_cpu);  

    int output_stride;
    if (!channel_last) {
      output_stride =
          (batch_idx * channels + offsetC) * output_height * output_width;
    } else {
      output_stride = batch_idx * output_height * output_width * channels;
    }
    output_gpu += output_stride;
    output_grad_cpu += output_stride;
    // CUDA_PRINT("index=%d output_stride=%d output_gpu=%1.1f output_grad_cpu=%1.1f", 
    //             index, output_stride, *output_gpu, *output_grad_cpu);

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pool_size;
        if (adaptive) {
          pool_size = static_cast<int>(ceil(static_cast<double>(input_height) /
                                            ksize_height)) *
                      static_cast<int>(
                          ceil(static_cast<double>(input_width) / ksize_width));
        } else {
          int hstart = ph * stride_height - padding_height;
          int wstart = pw * stride_width - padding_width;
          int hend = min(hstart + ksize_height, input_height);
          int wend = min(wstart + ksize_width, input_width);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          pool_size = exclusive ? (hend - hstart) * (wend - wstart)
                                : ksize_height * ksize_width;
        }
        // CUDA_PRINT("index=%d ph=%d pw=%d pool_size=%d", 
        //             index, ph, pw, pool_size);
        
        int output_sub_idx = channel_last
                                 ? (ph * output_width + pw) * channels + offsetC
                                 : ph * output_width + pw;
        pool_process.compute(input_cpu, output_gpu[output_sub_idx],
                             output_grad_cpu[output_sub_idx],
                             static_cast<T>(1.0 / pool_size), &gradient);
        // CUDA_PRINT("index=%d output_sub_idx=%d input_cpu=%1.1f output_gpu[output_sub_idx]=%1.1f output_grad_cpu[output_sub_idx]=%1.1f gradient=%1.1f", 
        //             index, output_sub_idx, input_cpu, output_gpu[output_sub_idx], output_grad_cpu[output_sub_idx], gradient);
      }
    }
    input_grad_cpu[index] = gradient;
    // CUDA_PRINT("index=%d input_grad_cpu[index]=%1.1f gradient=%1.1f", 
    //                 index, input_grad_cpu[index], gradient);
  }
}

template <typename T>
class Pool2dGradFunctor<AvgPoolGrad<T>, T> {
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
                  AvgPoolGrad<T> pool_process,
                  const bool exclusive, 
                  const bool adaptive) {
    printf("adaptive = %d\n", adaptive);

    const int input_numel = batch_size * input_channels * input_height * input_width;
    const int output_numel = batch_size * input_channels * output_height * output_width;

    T*input_gpu, *output_gpu, *output_grad_gpu, *input_grad_gpu;

    hipMalloc((void **)&input_gpu, input_numel * sizeof(T));
    hipMalloc((void **)&output_gpu, output_numel * sizeof(T));
    hipMalloc((void **)&output_grad_gpu, output_numel * sizeof(T));
    hipMalloc((void **)&input_grad_gpu, input_numel * sizeof(T));
    hipMemcpy(input_gpu, input_cpu, input_numel * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(output_gpu, output_cpu, output_numel * sizeof(T), hipMemcpyHostToDevice);
    hipMemcpy(output_grad_gpu, output_grad_cpu, output_numel * sizeof(T), hipMemcpyHostToDevice);

    int nthreads = batch_size * input_channels * input_height * input_width;
    int blocks = (nthreads + 256 - 1) / 256;
    dim3 threads(256, 1);
    dim3 grid(blocks, 1);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(KernelPool2DGrad<AvgPoolGrad<T>, T>), 
        dim3(grid), dim3(threads), 0, 0,
        nthreads, input_gpu, output_gpu, output_grad_gpu, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        pool_process, exclusive, adaptive, input_grad_gpu, false);

    hipMemcpy(input_grad_cpu, input_grad_gpu, input_numel * sizeof(T), hipMemcpyDeviceToHost);
  }
};

template class Pool2dGradFunctor<AvgPoolGrad<float>, float>;
