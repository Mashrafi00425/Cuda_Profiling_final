// Lab 2 - Vector addition, CUDA-event kernel timing
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);    \
        return EXIT_FAILURE;                                     \
    }                                                            \
} while (0)

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_a(n, 100.0f);
    std::vector<float> h_b(n, 200.0f);
    std::vector<float> h_c(n, 0.0f);
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // Warm-up before the timed run.
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    if (h_c[100] != 300.0f) {
        fprintf(stderr, "Incorrect result: c[100]=%f\n", h_c[100]);
        return EXIT_FAILURE;
    }
    printf("Elements: %d\n", n);
    printf("Kernel time: %.6f ms\n", elapsedMs);
    printf("c[100] = %.6f\n", h_c[100]);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return EXIT_SUCCESS;
}
