#include "cuda_utils.h"
#include "matrix_mult.h"

template <typename T>
void multiplyMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB) {
    size_t byteSizeA = numRowsA * numColsA * sizeof(T);
    size_t byteSizeB = numColsA * numColsB * sizeof(T);
    size_t byteSizeC = numRowsA * numColsB * sizeof(T);

    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));

    // Copy input matrices from host to device
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    const T alpha = 1.0;
    const T beta = 0.0;

    // Perform matrix multiplication using cuBLAS
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublasHandle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 numColsB, numRowsA, numColsA,
                                 &alpha,
                                 deviceMatrixB, numColsB,
                                 deviceMatrixA, numColsA,
                                 &beta,
                                 deviceResultMatrix, numColsB));
    } else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(cublasHandle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 numColsB, numRowsA, numColsA,
                                 &alpha,
                                 deviceMatrixB, numColsB,
                                 deviceMatrixA, numColsA,
                                 &beta,
                                 deviceResultMatrix, numColsB));
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float and double types are supported");
    }

    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}

// Explicit instantiations
template void multiplyMatricesOnGPU<float>(const float*, const float*, float*, int, int, int);
template void multiplyMatricesOnGPU<double>(const double*, const double*, double*, int, int, int);
