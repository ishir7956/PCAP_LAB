#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void parallelSelectionSort(int arr[], int n, int rank, int size, MPI_Comm comm) {
    int localN = n / size; // Number of elements per process
    int *localArr = (int *)malloc(localN * sizeof(int));

    // Scatter the input array to all processes
    MPI_Scatter(arr, localN, MPI_INT, localArr, localN, MPI_INT, 0, comm);

    // Perform local selection sort
    for (int i = 0; i < localN; ++i) {
        int minIndex = i;
        for (int j = i + 1; j < localN; ++j) {
            if (localArr[j] < localArr[minIndex]) {
                minIndex = j;
            }
        }
        if (minIndex != i) {
            int temp = localArr[i];
            localArr[i] = localArr[minIndex];
            localArr[minIndex] = temp;
        }
    }

    // Gather the sorted subarrays back to process 0
    MPI_Gather(localArr, localN, MPI_INT, arr, localN, MPI_INT, 0, comm);

    free(localArr);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 20; // Total number of elements
    int *arr = NULL;

    if (rank == 0) {
        // Only process 0 initializes the array
        arr = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; ++i) {
            arr[i] = rand() % 100;
        }
    }

    // Perform parallel selection sort
    parallelSelectionSort(arr, n, rank, size, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sorted Array:\n");
        for (int i = 0; i < n; ++i) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        free(arr);
    }

    MPI_Finalize();

    return 0;
}
