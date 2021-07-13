#include <stdio.h>
#include <iostream>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

template <typename T>
void TestMemcpy() {
    const size_t numel = 16;
    int * cpu_indata = new int[numel];
    int * cpu_outdata = new int[numel];
    for (unsigned i = 0; i < numel; i++) {
        cpu_indata[i] = i;
    }
    for (unsigned i = 0; i < numel; i++) {
        cpu_outdata[i] = 0;
    }

    std::cout << "cpu_indata = [";
    for (unsigned i = 0; i < numel; i++) {
        std::cout << cpu_indata[i] << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "cpu_outdata = [";
    for (unsigned i = 0; i < numel; i++) {
        std::cout << cpu_outdata[i] << ", ";
    }
    std::cout << "]" << std::endl;

    int *gpu_data;
#ifdef __HIPCC__
    hipMalloc((void**)&gpu_data, numel * sizeof(T));
    hipMemcpy(gpu_data, cpu_indata, numel * sizeof(T), hipMemcpyHostToDevice);

    hipDeviceSynchronize();

    hipMemcpy(cpu_outdata, gpu_data, numel * sizeof(T), hipMemcpyDeviceToHost);
#else
    cudaMalloc((void**)&gpu_data, numel * sizeof(T));
    cudaMemcpy(gpu_data, cpu_indata, numel * sizeof(T), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_outdata, gpu_data, numel * sizeof(T), cudaMemcpyDeviceToHost);
#endif

    std::cout << "cpu_outdata = [";
    for (unsigned i = 0; i < numel; i++) {
        std::cout << cpu_outdata[i] << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    TestMemcpy<int16_t>();
    return 0;
}
