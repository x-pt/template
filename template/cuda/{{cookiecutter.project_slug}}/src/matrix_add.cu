#include "cuda_utils.h"  // Custom CUDA utilities for error checking, etc.
#include "matrix_add.h"  // Header file for this matrix addition module

// Namespace to encapsulate CUDA kernel functions
namespace cuda_kernel {

// CUDA Kernel: Adds two matrices element-wise on the GPU
// Each thread computes a single element of the result matrix
// Parameters:
// - matrixA: Device pointer to the input matrix A
// - matrixB: Device pointer to the input matrix B
// - resultMatrix: Device pointer to the output result matrix
// - numRows: Number of rows in the matrices
// - numCols: Number of columns in the matrices
template <typename T>
__global__ void addMatricesKernel(const T* matrixA, const T* matrixB, T* resultMatrix, int numRows, int numCols) {
    // Calculate the row and column indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within valid matrix bounds
    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        resultMatrix[index] = matrixA[index] + matrixB[index];  // Perform element-wise addition
    }
}

}  // namespace cuda_kernel

// C++ Function: Handles matrix addition on the GPU
// Transfers matrices from the host (CPU) to the device (GPU), performs the computation,
// and then copies the result back to the host.
// Parameters:
// - hostMatrixA: Pointer to matrix A on the host (CPU)
// - hostMatrixB: Pointer to matrix B on the host (CPU)
// - hostResultMatrix: Pointer to the result matrix on the host (CPU)
// - numRows: Number of rows in the matrices
// - numCols: Number of columns in the matrices
template <typename T>
void addMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix, int numRows, int numCols) {
    // Calculate the size of the matrices in bytes
    size_t matrixSizeBytes = numRows * numCols * sizeof(T);

    // Device (GPU) memory pointers
    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    // Allocate memory on the device (GPU)
    CUDA_CHECK(cudaMalloc(&deviceMatrixA, matrixSizeBytes));       // Allocate memory for matrix A
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, matrixSizeBytes));       // Allocate memory for matrix B
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, matrixSizeBytes));  // Allocate memory for the result matrix

    // Copy input matrices from host (CPU) to device (GPU)
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, matrixSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, matrixSizeBytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions for launching the kernel
    dim3 threadsPerBlock(16, 16);  // Each block contains 16x16 threads
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);  // Calculate number of blocks required

    // Launch the CUDA kernel to add the matrices on the device
    cuda_kernel::addMatricesKernel<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix, numRows, numCols);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize the device to ensure kernel execution is complete
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result matrix from device (GPU) back to host (CPU)
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, matrixSizeBytes, cudaMemcpyDeviceToHost));

    // Free the allocated memory on the device
    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}

// Explicit template instantiations for float and double types
template void addMatricesOnGPU<float>(const float*, const float*, float*, int, int);
template void addMatricesOnGPU<double>(const double*, const double*, double*, int, int);
