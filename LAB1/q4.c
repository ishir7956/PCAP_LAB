#include "mpi.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char input_string[] = "HELLO";
    int str_length = strlen(input_string);

    if (rank == 0) {
        printf("Original string: %s\n", input_string);
    }

    int char_index = rank % str_length;

    if (input_string[char_index] >= 'A' && input_string[char_index] <= 'Z') {
        input_string[char_index] += 'a' - 'A';
    } else if (input_string[char_index] >= 'a' && input_string[char_index] <= 'z') {
        input_string[char_index] += 'A' - 'a';
    }

    printf("Modified string for Process %d: %s\n", rank, input_string);
    MPI_Finalize();
    return 0;
}
