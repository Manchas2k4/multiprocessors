#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	int iam, total;
	
	#pragma omp parallel 
	{
		total = omp_get_num_threads();
		iam = omp_get_thread_num();
		printf("Hola mundo, soy el %i de %i threads\n", iam, total);
	}
	return 0;
}
