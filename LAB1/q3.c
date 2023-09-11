#include "mpi.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int operand1 = 24;
    int operand2 = 8;
    int result = 0;

    if (size < 4) {
        if (rank == 0) {
            printf("Error: At least 4 processes required for this calculator.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        result = operand1 + operand2;
        printf("Operand 1: %d\n", operand1);
        printf("Operand 2: %d\n", operand2);
        printf("Result of addition: %d\n", result);
    }

    if (rank == 1) {
        result = operand1 - operand2;
        printf("Result of subtraction: %d\n", result);
    }

    if (rank == 2) {
        result = operand1 * operand2;
        printf("Result of multiplication: %d\n", result);
    }

    if (rank == 3) {
        if (operand2 != 0) {
            result = operand1 / operand2;
            printf("Result of division: %d\n", result);
        } else {
            printf("Error: Division by zero is not allowed.\n");
        }
    }
    MPI_Finalize();
    return 0;
}
