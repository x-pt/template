#include "cuda_utils.h"   // Custom CUDA utility functions and macros for error checking
#include "matrix_mult.h"   // Header for this matrix multiplication module

// Function to perform matrix multiplication on the GPU using cuBLAS
// This function transfers the input matrices from the host (CPU) to the device (GPU),
// executes the matrix multiplication on the GPU, and retrieves the result back to the host.
// Parameters:
// - hostMatrixA: Pointer to the first matrix (A) on the host (CPU)
// - hostMatrixB: Pointer to the second matrix (B) on the host (CPU)
// - hostResultMatrix: Pointer to the result matrix (C) on the host (CPU)
// - numRowsA: Number of rows in matrix A
// - numColsA: Number of columns in matrix A (and rows in matrix B)
// - numColsB: Number of columns in matrix B
template <typename T>
void multiplyMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB) {
    // Calculate the size of matrices A, B, and C in bytes
    size_t byteSizeA = numRowsA * numColsA * sizeof(T);
    size_t byteSizeB = numColsA * numColsB * sizeof(T);
    size_t byteSizeC = numRowsA * numColsB * sizeof(T);

    // Device (GPU) memory pointers for matrices A, B, and result matrix C
    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    // Allocate memory for matrices on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));  // Allocate memory for matrix A on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));  // Allocate memory for matrix B on the GPU
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));  // Allocate memory for result matrix C on the GPU

    // Copy matrices A and B from the host (CPU) to the device (GPU)
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    // Create a cuBLAS handle for matrix multiplication
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Define alpha and beta scalars for the matrix multiplication: C = alpha * A * B + beta * C
    const T alpha = 1.0;
    const T beta = 0.0;

    // Perform matrix multiplication using cuBLAS based on the type of T (float or double)
    // For float: Use cublasSgemm (single precision)
    if constexpr (std::is_same_v<T, float>) {
        CUBLAS_CHECK(cublasSgemm(cublasHandle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition for both matrices
                                 numColsB, numRowsA, numColsA,  // Dimensions of matrices
                                 &alpha,  // Scalar alpha
                                 deviceMatrixB, numColsB,  // Matrix B in device memory
                                 deviceMatrixA, numColsA,  // Matrix A in device memory
                                 &beta,  // Scalar beta
                                 deviceResultMatrix, numColsB));  // Result matrix C in device memory
    }
    // For double: Use cublasDgemm (double precision)
    else if constexpr (std::is_same_v<T, double>) {
        CUBLAS_CHECK(cublasDgemm(cublasHandle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,  // No transposition for both matrices
                                 numColsB, numRowsA, numColsA,  // Dimensions of matrices
                                 &alpha,  // Scalar alpha
                                 deviceMatrixB, numColsB,  // Matrix B in device memory
                                 deviceMatrixA, numColsA,  // Matrix A in device memory
                                 &beta,  // Scalar beta
                                 deviceResultMatrix, numColsB));  // Result matrix C in device memory
    }
    // If neither float nor double, throw a compile-time error
    else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Only float and double types are supported for matrix multiplication");
    }

    // Copy the result matrix from the device (GPU) back to the host (CPU)
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    // Clean up: Destroy cuBLAS handle and free the allocated GPU memory
    CUBLAS_CHECK(cublasDestroy(cublasHandle));  // Destroy cuBLAS context
    CUDA_CHECK(cudaFree(deviceMatrixA));  // Free memory for matrix A
    CUDA_CHECK(cudaFree(deviceMatrixB));  // Free memory for matrix B
    CUDA_CHECK(cudaFree(deviceResultMatrix));  // Free memory for result matrix C
}

// Explicit template instantiations for float and double types
template void multiplyMatricesOnGPU<float>(const float*, const float*, float*, int, int, int);
template void multiplyMatricesOnGPU<double>(const double*, const double*, double*, int, int, int);
