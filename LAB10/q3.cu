#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

// Swap function to exchange two elements
__device__ void swap(int* array, int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

// Odd-even transposition sort kernel
__global__ void oddEvenSort(int* array, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;
    int phase, other;

    for (phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {
            if (tid % 2 == 0 && tid + 1 < n) {
                if (array[tid] > array[tid + 1]) {
                    swap(array, tid, tid + 1);
                }
            }
        } else {
            if (tid % 2 != 0 && tid + 1 < n) {
                if (array[tid] > array[tid + 1]) {
                    swap(array, tid, tid + 1);
                }
            }
        }
        __syncthreads();
    }
}

int main() {
    int n = 10; // Adjust the number of elements as needed
    int* h_array = (int*)malloc(n * sizeof(int));
    int* d_array;

    // Initialize the array with random values
    for (int i = 0; i < n; i++) {
        h_array[i] = rand() % 1000;
    }

    cudaMalloc((void**)&d_array, n * sizeof(int));
    cudaMemcpy(d_array, h_array, n * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Perform odd-even transposition sort on the GPU
    for (int i = 0; i < n; i++) {
        oddEvenSort<<<numBlocks, BLOCK_SIZE>>>(d_array, n);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the sorted array
    printf("Sorted array:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    free(h_array);
    cudaFree(d_array);

    return 0;
}
