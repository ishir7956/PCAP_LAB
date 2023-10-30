#include <stdio.h>
#include<cuda.h>

#define N 4 
#define BLOCK_SIZE 2

// Kernel to perform matrix multiplication
__global__ void matrixMultiply(int *A, int *B, int *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    for (int i = 0; i < N; i++) {
        sum += A[row * N + i] * B[i * N + col];
    }

    C[row * N + col] = sum;
}

int main() {
    int A[N][N], B[N][N], C[N][N]; // Input and output matrices

    // Initialize matrices A and B

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = i * N + j;
            B[i][j] = j * N + i;
        }
    }

    int *d_A, *d_B, *d_C; // Device matrices

    // Allocate memory on the GPU
    cudaMalloc((void **)&d_A, N * N * sizeof(int));
    cudaMalloc((void **)&d_B, N * N * sizeof(int));
    cudaMalloc((void **)&d_C, N * N * sizeof(int));

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE, N / BLOCK_SIZE);

    // Launch the kernel
    matrixMultiply<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Copy the result matrix C from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the result matrix C
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;
}
