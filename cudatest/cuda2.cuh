
#pragma once

#include <cuda_runtime.h>

// CUDA kernel to add two matrices
__global__ void mulMatrices(int *A, int *B, int *C, int n);