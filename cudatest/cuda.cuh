
#pragma once

#include <cuda_runtime.h>

// CUDA kernel to add two matrices
__global__ void addMatrices(int *A, int *B, int *C, int n);