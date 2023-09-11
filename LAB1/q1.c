
#include "mpi.h"
#include <stdio.h>
#include <math.h>
int main(int argc, char *argv[])
{
	int rank,size;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int x=5;
	printf("The rank is %d and the power value is: %d\n", rank, (int)pow(x, rank));
	MPI_Finalize();
	return 0;
}