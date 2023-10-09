#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void replaceRowWithPowers(float* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int power;
    if(row == 0){
    	power = 1;
    }
    else if(row == 1){
    	power = 2;
    }
    else{
    	power = 3;
    }

    if (row < rows) {
        for (int i = 0; i < cols; i++) {
            matrix[row * cols + i] = powf(matrix[row * cols + i], power);
        }
    }
}

int main() {
    int M =3;
    int N=3;

    float* matrix = (float*)malloc(M * N * sizeof(float));

    printf("Enter the matrix elements:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%f", &matrix[i * N + j]);
        }
    }

    float* d_matrix;
    cudaMalloc((void**)&d_matrix, M * N * sizeof(float));
    cudaMemcpy(d_matrix, matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    replaceRowWithPowers<<<blocksPerGrid, threadsPerBlock>>>(d_matrix, M, N);
    cudaMemcpy(matrix, d_matrix, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Modified Matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", matrix[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_matrix);
    free(matrix);

    return 0;
}


