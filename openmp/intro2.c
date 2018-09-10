#include <stdio.h>
#include <omp.h>

int main(int argc, char* argv[]) {
	#pragma omp parallel
	{
		int total = omp_get_num_threads();
		int iam = omp_get_thread_num();
		printf("Hello world!! I am the thread %i from %i threads\n", iam, total);
	}
	return 0;
}
