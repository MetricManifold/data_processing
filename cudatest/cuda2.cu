
#include "cuda.cuh"


// CUDA kernel to add two matrices
__global__ void mulMatrices(int *A, int *B, int *C, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i * n + j;

	if (i < n && j < n) {
		C[index] = A[index] + B[index];
	}
}

