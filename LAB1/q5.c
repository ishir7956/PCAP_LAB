#include "mpi.h"
#include <stdio.h>

int factorial(int n) 
{
    if (n == 0 || n == 1)
        return 1;
    else
        return n * factorial(n - 1);
}

int fibonacci(int n) 
{
    if (n <= 0)
        return 0;
    else if (n == 1)
        return 1;
    else
        return fibonacci(n - 1) + fibonacci(n - 2);
}

int main(int argc, char *argv[]) 
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int result;
    if (rank % 2 == 0) 
    {
        result = factorial(rank);
        printf("Rank %d: Factorial of rank is %d\n", rank, result);
    } 
    else 
    {
        result = fibonacci(rank);
        printf("Rank %d: Fibonacci number of rank is %d\n", rank, result);
    }
    MPI_Finalize();

    return 0;
}
