


#include <iostream>

#include "cuda.cuh"
#include "cuda2.cuh"
#include "header.h"

#define N 3  // Define the size of the matrices

int main() {
	int A[N][N], B[N][N], C[N][N];
	int *d_A, *d_B, *d_C;
	int size = N * N * sizeof(int);

	// Initialize matrices A and B
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			A[i][j] = i + j;
			B[i][j] = i - j;
		}
	}

	// Allocate memory on the device
	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	// Copy matrices A and B to the device
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	// Define the number of threads per block and the number of blocks per grid
	dim3 threadsPerBlock(N, N);
	dim3 blocksPerGrid(1, 1);

	// Launch the CUDA kernel
	addMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
	mulMatrices<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

	// Copy the result matrix C back to the host
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	print_matrix(reinterpret_cast<double**>(C), N);

	// Free the allocated memory on the device
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}