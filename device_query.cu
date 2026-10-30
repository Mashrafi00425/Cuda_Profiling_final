// Lab 2 - CUDA device inventory
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);    \
        return EXIT_FAILURE;                                     \
    }                                                            \
} while (0)

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    printf("CUDA device count: %d\n", count);
    if (count == 0) {
        fprintf(stderr, "No CUDA device detected.\n");
        return EXIT_FAILURE;
    }

    int runtimeVersion = 0;
    int driverVersion = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    printf("CUDA runtime version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 1000) / 10);
    printf("CUDA driver version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 1000) / 10);

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Global memory: %.2f MiB\n",
               static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0));
        printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Maximum thread dimensions: %d x %d x %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Maximum grid dimensions: %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("GPU clock rate: %.0f MHz\n", prop.clockRate / 1000.0);
        printf("Memory clock rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
        printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    }
    return EXIT_SUCCESS;
}
