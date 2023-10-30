#include <stdio.h>

#define N 4  
#define M 3  
#define TILE_SIZE 2  


__global__ void convolution(int *input, int *mask, int *output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int maskRow = M - 1 - i;  // Flip mask vertically
            int maskCol = M - 1 - j;  // Flip mask horizontally
            int r = row + maskRow - M / 2;
            int c = col + maskCol - M / 2;

            if (r >= 0 && r < N && c >= 0 && c < N) {
                sum += input[r * width + c] * mask[i * M + j];
            }
        }
    }

    output[row * width + col] = sum;
}

int main() {
    int input[N][N], mask[M][M], output[N][N];  
    printf("Input Matrix: \n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            input[i][j] = i * N + j;
            printf("%d ", input[i][j]);
        }
        printf("\n");
    }
    printf("Mask:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            mask[i][j] = 1;
            printf("%d ", mask[i][j]);
        }
        printf("\n");
    }

    int *d_input, *d_mask, *d_output;  

    cudaMalloc((void **)&d_input, N * N * sizeof(int));
    cudaMalloc((void **)&d_mask, M * M * sizeof(int));
    cudaMalloc((void **)&d_output, N * N * sizeof(int));

    cudaMemcpy(d_input, input, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, M * M * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid(N / TILE_SIZE, N / TILE_SIZE);
    convolution<<<dimGrid, dimBlock>>>(d_input, d_mask, d_output, N);

    cudaMemcpy(output, d_output, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);

    printf("Output Array:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", output[i][j]);
        }
        printf("\n");
    }

    return 0;
}
