#include <iostream>
#include <stdio.h>
#include <vector>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

struct miopenWorkspace {
    explicit miopenWorkspace(size_t size) : size(size), data(nullptr) {
      hipMalloc(&data, size);
    }
    miopenWorkspace(const miopenWorkspace&) = delete;
    miopenWorkspace(miopenWorkspace&&) = default;
    miopenWorkspace& operator=(miopenWorkspace&&) = default;
    ~miopenWorkspace() {
      if (data) {
        hipFree(data);
      }
    }
    size_t size;
    void* data;
};

int main(void)
{
    // input
    const int batch_size = 1;
    const int input_channels = 3;
    const int input_height = 1;
    const int input_width = 1;
    // filter
    const int output_channels = 3;
    const int groups = 3;
    const int kernel_h = 1;
    const int kernel_w = 1;
    // attr
    const std::vector<int> conv_stride = {1, 1}; // height, width
    const std::vector<int> conv_padding = {1, 1, 2, 0}; // top, bottom, left, right
    const std::vector<int> conv_dilation = {1, 1}; // height, width
    // out
    const int output_height = static_cast<int>((input_height + conv_padding[0] + conv_padding[1] - (conv_dilation[0] * (kernel_h - 1) + 1)) / conv_stride[0] + 1);
    const int output_width = static_cast<int>((input_width + conv_padding[2] + conv_padding[3] - (conv_dilation[1] * (kernel_w - 1) + 1)) / conv_stride[1] + 1);
    // numel
    const int input_numel = batch_size * input_channels * input_height * input_width;
    const int output_numel = batch_size * output_channels * output_height * output_width;
    const int filter_numel = output_channels * input_channels/groups * kernel_h * kernel_w;
    // dims
    std::vector<int> input_dims{batch_size, input_channels, input_height, input_width};
    std::vector<int> output_dims{batch_size, output_channels, output_height, output_width};
    std::vector<int> filter_dims{output_channels, input_channels/groups, kernel_h, kernel_w};
    // std::vector<int> input_strides(4);
    // input_strides[3] = 1;
    // for (int i = 2; i >= 0; i--) {
    //    input_strides[i] = input_strides[i + 1] * input_dims[i + 1];
    // }
    // std::vector<int> output_strides(4);
    // output_strides[3] = 1;
    // for (int i = 2; i >= 0; i--) {
    //     output_strides[i] = output_strides[i + 1] * output_dims[i + 1];
    // }
    // std::vector<int> pads{conv_padding, conv_padding, conv_padding, conv_padding};
    // std::vector<int> strides{conv_stride, conv_stride, conv_stride, conv_stride};
    // std::vector<int> dilations{conv_dilation, conv_dilation, conv_dilation, conv_dilation};

    // feed data
    std::vector<float> input_data(input_numel);
    for (size_t i = 0; i < input_numel; ++i) {
      input_data[i] = 1.0;
    }
    std::vector<float> filter_data(filter_numel);
    for (size_t i = 0; i < filter_numel; ++i) {
      filter_data[i] = 1.0;
    }
    std::vector<float> output_data(output_numel);

    miopenHandle_t handle;
    miopenCreate(&handle);

    miopenTensorDescriptor_t idesc, odesc, wdesc;
    miopenConvolutionDescriptor_t cdesc;

    miopenCreateTensorDescriptor(&idesc);
    miopenCreateTensorDescriptor(&odesc);
    miopenCreateTensorDescriptor(&wdesc);
    miopenCreateConvolutionDescriptor(&cdesc);

    miopenSet4dTensorDescriptor(idesc, miopenFloat, batch_size, input_channels, input_height, input_width);
    miopenSet4dTensorDescriptor(odesc, miopenFloat, batch_size, output_channels, output_height, output_width);
    miopenSet4dTensorDescriptor(wdesc, miopenFloat, output_channels, input_channels/groups, kernel_h, kernel_w);
    miopenInitConvolutionDescriptor(cdesc, miopenConvolution, conv_padding, conv_padding, conv_stride, conv_stride, conv_dilation, conv_dilation);
    miopenSetConvolutionGroupCount(cdesc, groups);

    // forward search
    size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(handle, wdesc, idesc, cdesc, odesc, &workspace_size);
    std::cout << "Get workspace_size: " << workspace_size << std::endl;
    void * workspace_ptr = nullptr;
    if(workspace_size > 0) {
        miopenWorkspace miopen_workspace(workspace_size);
        workspace_ptr = miopen_workspace.data;
    }
    // alloc data on GPU
    float *input_data_gpu, *output_data_gpu, *filter_data_gpu;
    hipMalloc((void **)&input_data_gpu, input_numel * sizeof(float));
    hipMalloc((void **)&output_data_gpu, output_numel * sizeof(float));
    hipMalloc((void **)&filter_data_gpu, filter_numel * sizeof(float));
    hipMemcpy(input_data_gpu, input_data.data(), input_numel * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(filter_data_gpu, filter_data.data(), filter_numel * sizeof(float), hipMemcpyHostToDevice);

    miopenConvFwdAlgorithm_t algo;
    miopenConvAlgoPerf_t returnedAlgorithm;
    const int requestedAlgorithmCount = 1;
    int returnedAlgorithmCount = 0;
    miopenFindConvolutionForwardAlgorithm(handle,
      idesc, input_data_gpu, wdesc, filter_data_gpu, cdesc, odesc, output_data_gpu,
      requestedAlgorithmCount, &returnedAlgorithmCount, &returnedAlgorithm,
      workspace_ptr, workspace_size, false);
    algo = returnedAlgorithm.fwd_algo;
    std::cout << "miopenConvFwdAlgorithm_t choose algo " << algo << std::endl;
    // forward
    float alpha = 1.0f;
    float beta = 0.0f;
    miopenConvolutionForward(handle, &alpha, idesc, input_data_gpu, wdesc, filter_data_gpu,
      cdesc, algo, &beta, odesc, output_data_gpu, workspace_ptr, workspace_size);
    // copy output data to CPU
    hipMemcpy(output_data.data(), output_data_gpu, output_numel * sizeof(float), hipMemcpyDeviceToHost);

    for (size_t i = 0; i < output_numel; ++i) {
      printf("output_data[%02d] = %5.1f\n", i, output_data[i]);
    }
    return 0;
}