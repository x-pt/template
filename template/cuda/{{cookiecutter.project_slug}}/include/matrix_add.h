#pragma once

#include <cuda_runtime.h>

__global__ void addMatricesKernel(const float* A, const float* B, float* C, int rows, int cols);
void addMatricesOnGPU(const float* A, const float* B, float* C, int rows, int cols);
