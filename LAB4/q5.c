#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void odd_even_sort(int arr[], int n, int phase) {
    int start = phase % 2 == 1 ? 1 : 0;
    int step = 2;

    for (int i = start; i < n - 1; i += step) {
        if (arr[i] > arr[i + 1]) {
            int temp = arr[i];
            arr[i] = arr[i + 1];
            arr[i + 1] = temp;
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int input_array[] = {4, 7, 1, 9, 2, 5, 8, 3, 6, 12, 15, 11, 19, 22, 25, 18, 13, 16, 21, 14, 17, 10, 24, 23, 20, 28, 26};
    int n = sizeof(input_array) / sizeof(input_array[0]);


    int local_size = n / world_size;
    int local_array[local_size];

    MPI_Scatter(input_array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    for (int phase = 0; phase < n; ++phase) {
        odd_even_sort(local_array, local_size, phase);

        int partner;

        if (phase % 2 == 0) {
            if (world_rank % 2 == 0) {
                partner = world_rank + 1;
            } else {
                partner = world_rank - 1;
            }
        } else {
            if (world_rank % 2 == 0) {
                partner = world_rank - 1;
            } else {
                partner = world_rank + 1;
            }
        }

        if (partner >= 0 && partner < world_size) {
            int send_buf[local_size];
            int recv_buf[local_size];
            
            MPI_Sendrecv(local_array, local_size, MPI_INT, partner, 0, recv_buf, local_size, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if ((world_rank < partner && phase % 2 == 0) || (world_rank > partner && phase % 2 == 1)) {
                for (int i = 0; i < local_size; ++i) {
                    if (local_array[i] > recv_buf[i]) {
                        int temp = local_array[i];
                        local_array[i] = recv_buf[i];
                        recv_buf[i] = temp;
                    }
                }
            } else {
                for (int i = 0; i < local_size; ++i) {
                    if (local_array[i] < recv_buf[i]) {
                        int temp = local_array[i];
                        local_array[i] = recv_buf[i];
                        recv_buf[i] = temp;
                    }
                }
            }
        }
    }

    MPI_Gather(local_array, local_size, MPI_INT, input_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Input array: ");
        for (int i = 0; i < n; ++i) {
            printf("%d ", input_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
