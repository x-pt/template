#include "cuda_utils.h"
#include "matrix_add.h"

namespace cuda_kernel {

template <typename T>
__global__ void addMatricesKernel(const T* matrixA, const T* matrixB, T* resultMatrix, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        resultMatrix[index] = matrixA[index] + matrixB[index];
    }
}

} // namespace cuda_kernel

template <typename T>
void addMatricesOnGPU(const T* hostMatrixA, const T* hostMatrixB, T* hostResultMatrix, int numRows, int numCols) {
    size_t matrixSizeBytes = numRows * numCols * sizeof(T);

    T *deviceMatrixA, *deviceMatrixB, *deviceResultMatrix;

    CUDA_CHECK(cudaMalloc(&deviceMatrixA, matrixSizeBytes));
    CUDA_CHECK(cudaMalloc(&deviceMatrixB, matrixSizeBytes));
    CUDA_CHECK(cudaMalloc(&deviceResultMatrix, matrixSizeBytes));

    // Copy input matrices from host to device
    CUDA_CHECK(cudaMemcpy(deviceMatrixA, hostMatrixA, matrixSizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceMatrixB, hostMatrixB, matrixSizeBytes, cudaMemcpyHostToDevice));

    // Define the grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cuda_kernel::addMatricesKernel<<<numBlocks, threadsPerBlock>>>(
        deviceMatrixA, deviceMatrixB, deviceResultMatrix, numRows, numCols);

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host memory
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, matrixSizeBytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}

// Explicit instantiations
template void addMatricesOnGPU<float>(const float*, const float*, float*, int, int);
template void addMatricesOnGPU<double>(const double*, const double*, double*, int, int);
