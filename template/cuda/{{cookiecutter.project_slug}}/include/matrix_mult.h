#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Function to perform matrix multiplication on the GPU using cuBLAS
template <typename T>
void multiplyMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix,
                           int numRowsA, int numColsA, int numColsB);

// Explicit instantiation declarations for multiplyMatricesOnGPU
extern template void multiplyMatricesOnGPU<float>(const float*, const float*, float*, int, int, int);
extern template void multiplyMatricesOnGPU<double>(const double*, const double*, double*, int, int, int);
