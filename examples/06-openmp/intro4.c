#include <stdio.h>
#include <omp.h>

#define N 7

int main(int argc, char* argv[]) {
	int i;

	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		for (i = 0; i < N; i++) {
			printf("Thread ID %i Iteration %i\n", id ,i);
		}
	}
	return 0;
}
