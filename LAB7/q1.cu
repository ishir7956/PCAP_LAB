#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__host__ __device__ void printMatrix(const char* string, int* A, int width) {
    printf("%s\n", string);
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++)
            printf("%d\t", A[i * width + j]);
        printf("\n");
    }
    printf("\n");
}

__host__ void clearMatrix(int* A, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++)
            A[i * width + j] = 0;
    }
}

__global__ void addMatrixKernel_1a(int* A, int* B, int* C, int width) {
    int row = threadIdx.y;
    for (int i = 0; i < width; i++) {
        C[row * width + i] = A[row * width + i] + B[row * width + i];
    }
}

__global__ void addMatrixKernel_1b(int* A, int* B, int* C, int width) {
    int col = threadIdx.x;
    for (int i = 0; i < width; i++) {
        C[i * width + col] = A[i * width + col] + B[i * width + col];
    }
}

__global__ void addMatrixKernel_1c(int* A, int* B, int* C, int width) {
    int col = threadIdx.x;
    int row = threadIdx.y;
    C[row * width + col] = A[row * width + col] + B[row * width + col];
}

void addMatrix(int* h_A, int* h_B, int* h_C, int width) {
    int* d_A, *d_B, *d_C;
    int size = width * width * sizeof(int);
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    dim3 dimBlock(1, 1, 1);
    dim3 dimGrid(1, 1, 1);
    dimBlock.x = 1;
    dimBlock.y = width;
    dimBlock.z = 1;
    addMatrixKernel_1a<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix("A + B (from 1a kernel): ", h_C, width);
    clearMatrix(h_C, width);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    dimBlock.x = width;
    dimBlock.y = 1;
    dimBlock.z = 1;
    addMatrixKernel_1b<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix("A + B (from 1b kernel): ", h_C, width);
    clearMatrix(h_C, width);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
    dimBlock.x = width;
    dimBlock.y = width;
    dimBlock.z = 1;
    addMatrixKernel_1c<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printMatrix("A + B (from 1c kernel): ", h_C, width);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int* A, *B, *C;
    int width;
    printf("Enter width: ");
    scanf("%d", &width);
    int size = width * width * sizeof(int);
    A = (int*)malloc(size);
    B = (int*)malloc(size);
    C = (int*)malloc(size);
    printf("Enter %d elements for A: ", (width * width));
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            scanf("%d", &A[i * width + j]);
        }
    }
    printf("Enter %d elements for B: ", (width * width));
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            scanf("%d", &B[i * width + j]);
        }
    }
    printMatrix("A:", A, width);
    printMatrix("B:", B, width);
    addMatrix(A, B, C, width);
    return 0;
}
