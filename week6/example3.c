#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	printf("SHARED\n");
	int x = 1;
	#pragma omp parallel shared(x) num_threads(2)
	{
		x++;
		if (x == 2) {
			printf("x = 2\n");
		}
		printf("In the parallel x is: %i\n", x);
	}
	printf("Outside the parallel x is: %i\n\n\n", x);
	
	x = 2;
	#pragma omp parallel private(x) num_threads(3)
	{
		// int x = 0;
		x++;
		printf("In the parallel x is: %i\n", x);
	}
	printf("Outside the parallel x is: %i\n\n\n", x);
	
	x = 2;
	#pragma omp parallel firstprivate(x) num_threads(3)
	{
		// int x = 2;
		x++;
		printf("In the parallel x is: %i\n", x);
	}
	printf("Outside the parallel x is: %i\n\n\n", x);
	return 0;
}
