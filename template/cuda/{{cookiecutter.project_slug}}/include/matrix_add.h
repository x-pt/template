#pragma once

#include <cuda_runtime.h>

namespace cuda_kernel {

// CUDA kernel for matrix addition
template <typename T>
__global__ void addMatricesKernel(const T* matrixA, const T* matrixB, T* resultMatrix, int numRows, int numCols);

} // namespace cuda_kernel

// Function to perform matrix addition on the GPU
template <typename T>
void addMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix, int numRows, int numCols);

// Explicit instantiation declarations for addMatricesOnGPU
extern template void addMatricesOnGPU<float>(const float*, const float*, float*, int, int);
extern template void addMatricesOnGPU<double>(const double*, const double*, double*, int, int);
