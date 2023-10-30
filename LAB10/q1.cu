#include <stdio.h>
#include <math.h>
#include <cuda.h>
#define FILTER_SIZE 7  // Filter size
#define DATA_SIZE 32 // Input data size

// Define the filter coefficients as constant memory
__constant__ float d_Filter[FILTER_SIZE];

// CUDA kernel for 1D convolution
__global__ void convolutionKernel(float* input, float* output, int dataSize) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < dataSize) {
        float result = 0.0f;
        int halfFilterSize = FILTER_SIZE / 2;

        for (int i = 0; i < FILTER_SIZE; i++) {
            int idx = tid + i - halfFilterSize;
            if (idx >= 0 && idx < dataSize) {
                result += input[idx] * d_Filter[i];
            }
        }

        output[tid] = result;
    }
}

int main() {
    float h_Input[DATA_SIZE];
    float h_Output[DATA_SIZE];
    float h_Filter[FILTER_SIZE] = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f}; // Example filter

    // Initialize input data
    for (int i = 0; i < DATA_SIZE; i++) {
        h_Input[i] = sinf(0.1f * i); // Example input data
    }

    // Copy filter coefficients to constant memory
    cudaMemcpyToSymbol(d_Filter, h_Filter, FILTER_SIZE * sizeof(float));

    float* d_Input, *d_Output;
    cudaMalloc((void**)&d_Input, DATA_SIZE * sizeof(float));
    cudaMalloc((void**)&d_Output, DATA_SIZE * sizeof(float));

    cudaMemcpy(d_Input, h_Input, DATA_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (DATA_SIZE + threadsPerBlock - 1) / threadsPerBlock;

    convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Input, d_Output, DATA_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Output, d_Output, DATA_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the result
    for (int i = 0; i < DATA_SIZE; i++) {
        printf("Output[%d] = %f\n", i, h_Output[i]);
    }

    cudaFree(d_Input);
    cudaFree(d_Output);

    return 0;
}
