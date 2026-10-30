// Lab 2 - Memory coalescing: same kernel with stride 1 versus stride 32
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err = (call);                                    \
    if (err != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error: %s (%s:%d)\n",             \
                cudaGetErrorString(err), __FILE__, __LINE__);    \
        return EXIT_FAILURE;                                     \
    }                                                            \
} while (0)

__global__ void strideReadKernel(const float* input, float* output,
                                 int n, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input[static_cast<size_t>(i) * stride] + 1.0f;
    }
}

int main(int argc, char** argv) {
    int onlyStride = 0;        // 0 means measure both strides
    int stressSeconds = 0;
    bool profile = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--stride") == 0 && i + 1 < argc) {
            onlyStride = atoi(argv[++i]);
            if (onlyStride != 1 && onlyStride != 32) {
                fprintf(stderr, "--stride must be 1 or 32\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--profile") == 0) {
            profile = true;
        } else if (strcmp(argv[i], "--stress-seconds") == 0 && i + 1 < argc) {
            stressSeconds = atoi(argv[++i]);
            if (stressSeconds < 1 || stressSeconds > 120) {
                fprintf(stderr, "--stress-seconds must be from 1 to 120\n");
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "Usage: %s [--stride 1|32] [--profile] "
                    "[--stress-seconds 5]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    const int n = 1 << 18;
    const int maxStride = 32;
    const size_t inputCount = static_cast<size_t>(n - 1) * maxStride + 1;
    const size_t inputBytes = inputCount * sizeof(float);
    const size_t outputBytes = static_cast<size_t>(n) * sizeof(float);
    std::vector<float> h_input(inputCount);
    std::vector<float> h_output(n);
    for (size_t i = 0; i < inputCount; ++i)
        h_input[i] = static_cast<float>(i % 101);

    float *d_input = NULL, *d_output = NULL;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), inputBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), outputBytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), inputBytes, cudaMemcpyHostToDevice));
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int strides[] = {1, 32};

    for (int s = 0; s < 2; ++s) {
        const int stride = strides[s];
        if (onlyStride && stride != onlyStride) continue;

        // Required untimed warm-up launch.
        strideReadKernel<<<blocks, threads>>>(d_input, d_output, n, stride);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (stressSeconds) {
            const auto start = std::chrono::steady_clock::now();
            do {
                for (int j = 0; j < 64; ++j) {
                    strideReadKernel<<<blocks, threads>>>(d_input, d_output, n, stride);
                }
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            } while (std::chrono::duration_cast<std::chrono::seconds>(
                         std::chrono::steady_clock::now() - start).count() < stressSeconds);
            printf("Stride %d stress kernel completed (%d seconds).\n",
                   stride, stressSeconds);
            continue;
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        const int repetitions = profile ? 1 : 100;
        CUDA_CHECK(cudaEventRecord(start));
        for (int j = 0; j < repetitions; ++j) {
            strideReadKernel<<<blocks, threads>>>(d_input, d_output, n, stride);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float totalMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&totalMs, start, stop));
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, outputBytes,
                              cudaMemcpyDeviceToHost));
        const float expected = h_input[static_cast<size_t>(100) * stride] + 1.0f;
        if (h_output[100] != expected) {
            fprintf(stderr, "Stride %d failed check (got %f, expected %f).\n",
                    stride, h_output[100], expected);
            return EXIT_FAILURE;
        }
        printf("Stride %d average kernel time: %.6f ms (%d timed runs, "
               "1 warm-up launch)\n", stride, totalMs / repetitions, repetitions);
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return EXIT_SUCCESS;
}
