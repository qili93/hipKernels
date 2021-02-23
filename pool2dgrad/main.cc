#include <iostream>
#include <stdio.h>

#include "pooling.h"

int main(void)
{
    // input_cpu
    const int batch_size = 1;
    const int input_channels = 1;
    const int input_height = 2;
    const int input_width = 2;
    // attr
    const int ksize_height = 1;
    const int ksize_width = 1;
    const int stride_height = 1;
    const int stride_width = 1;
    const int padding_height = 0;
    const int padding_width = 0;
    const std::string padding_algorithm = "EXPLICIT";
    // attr
    const bool ceil_mode = false;
    const bool exclusive = true;
    const bool adaptive = false;
    const bool global_pooling = false;
    const std::string pooling_t = "avg";
    const std::string data_format = "NCHW";
    const bool channel_last = (data_format == "NHWC");
    // numel
    const int output_height = 1;
    const int output_width = 1;
    const int input_numel = batch_size*input_channels*input_height*input_width;
    const int output_numel = batch_size*input_channels*output_height*output_width;

    float * input_cpu = new float[input_numel];
    float * output_cpu = new float[output_numel];
    float * output_grad_cpu = new float[output_numel];
    float * input_grad_cpu = new float[input_numel];

    for (int i = 0; i < input_numel; ++i) {input_cpu[i] = 1.0; input_grad_cpu[i] = 0.0;}
    for (int i = 0; i < output_numel; ++i) {output_cpu[i] = 1.0; output_grad_cpu[i] = 1.0;}

    // for(int i = 0; i < input_numel; ++i) {
    //     printf("input_cpu[%d] = %1.1f\n", i, input_cpu[i]);
    // }
    // for(int i = 0; i < output_numel; ++i) {
    //   printf("output_cpu[%d] = %1.1f\n", i, output_cpu[i]);
    // }
    // for(int i = 0; i < output_numel; ++i) {
    //   printf("output_grad_cpu[%d] = %1.1f\n", i, output_grad_cpu[i]);
    // }

    printf("adaptive = %d\n", adaptive);
    Pool2dGradFunctor<AvgPoolGrad<float>, float>  pool2d_backward;
    AvgPoolGrad<float> pool_process;

    pool2d_backward(batch_size, input_channels, input_height, input_width, output_height, output_width,
                    ksize_height, ksize_width, stride_height, stride_width, padding_height, padding_width,
                    input_cpu, output_cpu, output_grad_cpu, input_grad_cpu, pool_process, exclusive, adaptive);

    // for (int i = 0; i < input_numel; ++i) {
    //     printf("input_grad_cpu[%d] = %1.1f\n", i, input_grad_cpu[i]);
    // }

    return 0;
}