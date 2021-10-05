#include <stdio.h>
#include <omp.h>

#define N 3

int main(int argc, char* argv[]) {
	int i;

	#pragma omp parallel private(i) num_threads(4)
	{
		int id = omp_get_thread_num();
		for (i = 0; i < N; i++) {
			printf("Thread ID %i Iteration %i\n", id ,i);
		}
	}
	return 0;
}
