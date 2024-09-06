#pragma once

#include <cuda_runtime.h>

__global__ void matrixAdd(const float* A, const float* B, float* C, int rows, int cols);
void matrixAddHost(const float* A, const float* B, float* C, int rows, int cols);
