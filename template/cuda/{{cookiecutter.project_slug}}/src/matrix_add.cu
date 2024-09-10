#include "cuda_utils.h"

// CUDA kernel for adding two matrices element-wise
__global__ void addMatricesKernel(const float* matrixA, const float* matrixB, float* resultMatrix, int numRows, int numCols) {
    // Calculate the global row and column indices for this thread
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if this thread is within the matrix bounds
    if (rowIndex < numRows && colIndex < numCols) {
        // Calculate the linear index for the current element
        int elementIndex = rowIndex * numCols + colIndex;
        // Perform element-wise addition
        resultMatrix[elementIndex] = matrixA[elementIndex] + matrixB[elementIndex];
    }
}

// Host function to set up and execute matrix addition on GPU
void addMatricesOnGPU(const float* hostMatrixA, const float* hostMatrixB, float* hostResultMatrix, int numRows, int numCols) {
    // Calculate total size of the matrices in bytes
    size_t matrixSizeBytes = numRows * numCols * sizeof(float);

    // Declare pointers for device (GPU) memory
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    // Allocate memory on the GPU
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

    // Launch the CUDA kernel
    addMatricesKernel<<<numBlocks, threadsPerBlock>>>(deviceMatrixA, deviceMatrixB, deviceResultMatrix, numRows, numCols);

    // Check for errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the result back to host memory
    CUDA_CHECK(cudaMemcpy(hostResultMatrix, deviceResultMatrix, matrixSizeBytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(deviceMatrixA));
    CUDA_CHECK(cudaFree(deviceMatrixB));
    CUDA_CHECK(cudaFree(deviceResultMatrix));
}
