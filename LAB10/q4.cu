#include <stdio.h>
#include <stdlib.h>
#include<cuda.h>

#define NUM_THREADS 256

__global__ void mergeSort(int* array, int* temp, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int step, start, end;

    for (step = 1; step < n; step *= 2) {
        for (start = 0; start < n - step; start += step * 2) {
            end = min(start + step * 2, n);
            int middle = min(start + step, n);
            int i = start, j = middle;
            for (int k = start; k < end; k++) {
                if (i < middle && (j >= end || array[i] <= array[j])) {
                    temp[k] = array[i];
                    i++;
                } else {
                    temp[k] = array[j];
                    j++;
                }
            }
            for (int k = start; k < end; k++) {
                array[k] = temp[k];
            }
        }
    }
}

void mergeSortWrapper(int* array, int n) {
    int* d_array;
    int* d_temp;

    cudaMalloc((void**)&d_array, n * sizeof(int));
    cudaMalloc((void**)&d_temp, n * sizeof(int));

    cudaMemcpy(d_array, array, n * sizeof(int), cudaMemcpyHostToDevice);

    mergeSort<<<1, NUM_THREADS>>>(d_array, d_temp, n);
    cudaDeviceSynchronize();

    cudaMemcpy(array, d_array, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_array);
    cudaFree(d_temp);
}

int main() {
    int n = 10; // The number of elements
    int* array = (int*)malloc(n * sizeof(int));

    // Initialize the array with unsorted values

    for (int i = 0; i < n; i++) {
        array[i] = rand() % 100;
    }

    mergeSortWrapper(array, n);

    // Print sorted array
    for (int i = 0; i < n; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");

    free(array);

    return 0;
}
