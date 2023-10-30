#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MASK_WIDTH 3
#define TILE_SIZE 16
#define N 512

__global__ void embossKernel(unsigned char* input, unsigned char* output) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = MASK_WIDTH / 2;
    int sum = 0;

    if (col >= offset && col < (N - offset) && row >= offset && row < (N - offset)) {
        for (int i = -offset; i <= offset; i++) {
            for (int j = -offset; j <= offset; j++) {
                sum += input[(row + i) * N + (col + j)] * (-1);
            }
        }
        output[row * N + col] = (unsigned char)(sum + 128);
    }
}

int main() {
    unsigned char *h_input, *h_output;
    unsigned char *d_input, *d_output;
    int size = N * N * sizeof(unsigned char);
    h_input = (unsigned char*)malloc(size);
    h_output = (unsigned char*)malloc(size);
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    embossKernel<<<dimGrid, dimBlock>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
