#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void processMatrix(int* A, int* B, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        int idx = row * N + col;

        // Check if the element is a non-border element
        if (row > 0 && row < M - 1 && col > 0 && col < N - 1) {
            int binary[32];
            int one[32];
            int i =0;
            int decimal = A[idx];
            while(decimal>0){
                binary[i] = decimal % 2;
                decimal = decimal/2;
                i++;
            }
            for(int j = i-1; j>=0; j--){
                one[j] = 1-binary[j];
            }
            int comp;
            i--;
            while(i>=0){
                
            }
        } else {
            B[idx] = A[idx];  // Keep the original value
        }
    }
}

int main() {
    int M, N;

    // Read the dimensions of the matrix from the user
    printf("Enter the number of rows (M): ");
    scanf("%d", &M);
    printf("Enter the number of columns (N): ");
    scanf("%d", &N);

    // Allocate memory for matrices A and B on the host
    int* A = (int*)malloc(M * N * sizeof(int));
    int* B = (int*)malloc(M * N * sizeof(int));

    // Read matrix A from the user
    printf("Enter the matrix elements of A (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            scanf("%d", &A[i * N + j]);
        }
    }

    // Allocate memory for matrices A and B on the device
    int* d_A, *d_B;
    cudaMalloc((void**)&d_A, M * N * sizeof(int));
    cudaMalloc((void**)&d_B, M * N * sizeof(int));

    // Copy matrix A from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch the CUDA kernel to process the matrix
    processMatrix<<<gridDim, blockDim>>>(d_A, d_B, M, N);

    // Copy matrix B from device to host
    cudaMemcpy(B, d_B, M * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the resulting matrix B
    printf("Matrix B (%dx%d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B);

    return 0;
}
