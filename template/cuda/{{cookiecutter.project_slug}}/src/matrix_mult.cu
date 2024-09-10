#include "cuda_utils.h"

// Function to perform matrix multiplication on GPU using cuBLAS
void multiplyMatricesOnGPU(const float* hostMatrixA, const float* hostMatrixB, float* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB) {
    // Calculate sizes in bytes for each matrix
    size_t byteSizeA = numRowsA * numColsA * sizeof(float);
    size_t byteSizeB = numColsA * numColsB * sizeof(float);
    size_t byteSizeC = numRowsA * numColsB * sizeof(float);

    // Declare pointers for device (GPU) memory
    float *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc(&deviceMatrixA, byteSizeA));
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, byteSizeB));
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, byteSizeC));

    // Copy input matrices from host to device
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, byteSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, byteSizeB, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t cublasHandle;
    CUBLAS_CHECK(cublasCreate(&cublasHandle));

    // Set up parameters for cublasSgemm
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication using cuBLAS
    CUBLAS_CHECK(cublasSgemm(cublasHandle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             numColsB, numRowsA, numColsA,
                             &alpha,
                             deviceMatrixB, numColsB,
                             deviceMatrixA, numColsA,
                             &beta,
                             deviceResultMatrix, numColsB));

    // Copy the result back to host memory
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, byteSizeC, cudaMemcpyDeviceToHost));

    // Clean up: Free GPU memory and destroy cuBLAS handle
    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
}
