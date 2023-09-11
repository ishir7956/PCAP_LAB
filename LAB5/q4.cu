#include <stdio.h>
#include <math.h>

// Kernel function to calculate sine of angles
__global__ void calculateSine(float *input, float *output, int numAngles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < numAngles) {
        output[index] = sinf(input[index]);
    }
}

int main() {
    int numAngles = 10;
    size_t dataSize = numAngles * sizeof(float);
    
    // Allocate memory for input and output arrays on the host
    float *angles = (float *)malloc(dataSize);
    float *sineValues = (float *)malloc(dataSize);
    
    // Initialize input array with angles in radians
    for (int i = 0; i < numAngles; ++i) {
        angles[i] = i * 0.1f; // Replace with your desired angle values
    }
    
    // Allocate memory for input and output arrays on the device
    float *d_angles, *d_sineValues;
    cudaMalloc((void **)&d_angles, dataSize);
    cudaMalloc((void **)&d_sineValues, dataSize);
    
    // Copy input array from host to device
    cudaMemcpy(d_angles, angles, dataSize, cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (numAngles + blockSize - 1) / blockSize;
    
    // Launch kernel to calculate sine values
    calculateSine<<<gridSize, blockSize>>>(d_angles, d_sineValues, numAngles);
    
    // Copy output array from device to host
    cudaMemcpy(sineValues, d_sineValues, dataSize, cudaMemcpyDeviceToHost);
    
    // Print the results
    printf("Angle (radians)\tSine Value\n");
    for (int i = 0; i < numAngles; ++i) {
        printf("%.2f\t\t%.4f\n", angles[i], sineValues[i]);
    }
    
    // Free allocated memory on the host and device
    free(angles);
    free(sineValues);
    cudaFree(d_angles);
    cudaFree(d_sineValues);
    
    return 0;
}
