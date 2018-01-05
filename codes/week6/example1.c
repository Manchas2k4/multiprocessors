#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	int total, id;
	
	#pragma omp parallel num_threads(4)
	{
		total = omp_get_num_threads();
		id = omp_get_thread_num();
		printf("¡Hola mundo! Soy el thread %i de %i\n", id, total);
	}
	return 0;
}
