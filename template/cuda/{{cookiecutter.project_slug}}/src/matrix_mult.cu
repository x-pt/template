#include "cuda_utils.h"

void matrixMultHost(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB) {
    size_t sizeA = rowsA * colsA * sizeof(float);
    size_t sizeB = colsA * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);

    float* d_A;
    float* d_B;
    float* d_C;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            colsB, rowsA, colsA,
                            &alpha,
                            d_B, colsB,
                            d_A, colsA,
                            &beta,
                            d_C, colsB));

    CUDA_CHECK(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(handle));
}
