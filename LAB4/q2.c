#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NUM_RECTANGLES 1000000

double f(double x) {
    return 4.0 / (1.0 + x*x);
}

int main(int argc, char** argv) {
    int rank, size;
    double local_sum = 0.0, global_sum = 0.0;
    double total_width = 1.0;
    double rectangle_width = total_width / NUM_RECTANGLES;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_num_rectangles = NUM_RECTANGLES / size;
    int start_index = rank * local_num_rectangles;
    int end_index = start_index + local_num_rectangles - 1;

    for (int i = start_index; i <= end_index; ++i) {
        double x = (i + 0.5) * rectangle_width;
        local_sum += f(x) * rectangle_width;
    }

    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Approximation of pi: %.8f\n", global_sum);
    }

    MPI_Finalize();

    return 0;
}
