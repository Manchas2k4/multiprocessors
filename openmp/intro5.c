#include <stdio.h>
#include <omp.h>

#define N 20

int main(int argc, char* argv[]) {
	int i;
	
	#pragma omp parallel for
	for (i = 0; i < N; i++) {
		int id = omp_get_thread_num();
		printf("Thread ID %i Iteration %i\n", id, i);
	}
	return 0;
}
