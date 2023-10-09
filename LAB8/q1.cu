#include <stdio.h>
#include <cuda_runtime.h>

#define N 4 
#define NNZ 5 

__global__ void csrMatrixVectorMul(const int* row_ptr, const int* col_indices, const float* values, const float* x, float* y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
   
    if (tid < N) {
        float dot_product = 0.0f;
        int row_start = row_ptr[tid];
        int row_end = row_ptr[tid + 1];
       
        for (int j = row_start; j < row_end; j++) {
            dot_product += values[j] * x[col_indices[j]];
        }
       
        y[tid] = dot_product;
    }
}

int main() {
    int row_ptr[N + 1] = {0, 2, 3, 4,5};
    int col_indices[NNZ] = {0, 2, 1, 3, 2};
    float values[NNZ] = {2.0, 3.0, 4.0, 1.0, 5.0};

    float x[N] = {1.0, 2.0, 3.0, 4.0};
    float y[N]; 
    int *d_row_ptr, *d_col_indices;
    float *d_values, *d_x, *d_y;

    cudaMalloc((void**)&d_row_ptr, (N + 1) * sizeof(int));
    cudaMalloc((void**)&d_col_indices, NNZ * sizeof(int));
    cudaMalloc((void**)&d_values, NNZ * sizeof(float));
    cudaMalloc((void**)&d_x, N * sizeof(float));
    cudaMalloc((void**)&d_y, N * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, col_indices, NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);

    csrMatrixVectorMul<<<grid, block>>>(d_row_ptr, d_col_indices, d_values, d_x, d_y);

    cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result vector:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", y[i]);
    }
    printf("\n");

    cudaFree(d_row_ptr);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}



