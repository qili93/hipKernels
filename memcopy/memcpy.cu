#include <stdio.h>
#include <iostream>
#include <vector>

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#endif

template <typename T>
void TestMemcpy() {
    const size_t numel = 16;
    std::vector<T> cpu_indata(numel, 5);
    std::vector<T> cpu_outdata(numel, 0);

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
    hipMemcpy(gpu_data, cpu_indata.data(), numel * sizeof(T), hipMemcpyHostToDevice);

    hipDeviceSynchronize();

    hipMemcpy(cpu_outdata.data(), gpu_data, numel * sizeof(T), hipMemcpyDeviceToHost);
#else
    cudaMalloc((void**)&gpu_data, numel * sizeof(T));
    cudaMemcpy(gpu_data, cpu_indata.data(), numel * sizeof(T), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaMemcpy(cpu_outdata.data(), gpu_data, numel * sizeof(T), cudaMemcpyDeviceToHost);
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
