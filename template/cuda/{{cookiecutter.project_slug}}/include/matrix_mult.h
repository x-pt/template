#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

void multiplyMatricesOnGPU(const float* A, const float* B, float* C, int rowsA, int colsA, int colsB);
