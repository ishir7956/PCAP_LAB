#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

const int BlockSize = 256;

__global__ void convolutionKernel(float* N, float* M, float* P, int width, int mask_width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_mask_width = mask_width / 2;
    float value = 0.0f;

    for (int j = 0; j < mask_width; j++) {
        int idx = tid - half_mask_width + j;
        if (idx >= 0 && idx < width) {
            value += N[idx] * M[j];
        }
    }

    P[tid] = value;
}

int main() {
    int width = 1000;            
    int mask_width = 5;         
    float* N, * M, * P;          
    float* d_N, * d_M, * d_P;           

    N = (float*)malloc(width * sizeof(float));
    M = (float*)malloc(mask_width * sizeof(float));
    P = (float*)malloc(width * sizeof(float));

    for (int i = 0; i < width; i++) {
        N[i] = static_cast<float>(i);
    }

    for (int i = 0; i < mask_width; i++) {
        M[i] = 1.0f;
    }

    cudaMalloc((void**)&d_N, width * sizeof(float));
    cudaMalloc((void**)&d_M, mask_width * sizeof(float));
    cudaMalloc((void**)&d_P, width * sizeof(float));

    cudaMemcpy(d_N, N, width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, mask_width * sizeof(float), cudaMemcpyHostToDevice);

    int numBlocks = (width + BlockSize - 1) / BlockSize;
    convolutionKernel<<<numBlocks, BlockSize>>>(d_N, d_M, d_P, width, mask_width);

    cudaMemcpy(P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    for (int i = 0; i < width; i++) {
        printf("P[%d] = %.2f\n", i, P[i]);
    }

    free(N);
    free(M);
    free(P);

    return 0;
}
