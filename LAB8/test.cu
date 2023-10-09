
#include <stdio.h>
#include <cuda_runtime.h>

#define M 4
#define N 4
#define BITS 32

__device__ int calculateOnesComplement(int num) {
    int binary = 0;

    // Converting decimal to binary representation
    int bit = 1;
    while (num > 0) {
        int lastBit = num & 1;
        binary |= ((lastBit == 0) ? bit : 0);
        bit <<= 1;
        num >>= 1;
    }

    return binary;
}

__global__ void onesComplement(int* A, int* B, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < rows - 1 && j > 0 && j < cols - 1) {
        int index = i * cols + j;

        B[index] = calculateOnesComplement(A[index]);
    } else {
        int index = i * cols + j;
        B[index] = A[index];
    }
}

int main() {
    int A[M][N] = {
        {1, 2, 3, 4},
        {6, 5, 8, 3},
        {2, 4, 10, 1},
        {9, 1, 2, 5}
    };

    int B[M][N];

    int* d_A;
    int* d_B;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    onesComplement<<<gridDim, blockDim>>>(d_A, d_B, M, N);

    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    printf("Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }

    return 0;
}
