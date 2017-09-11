#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

#define N 20

int main(int argc, char* argv[]) {
	int i;

	#pragma omp parallel private(i)
	{
		int id = omp_get_thread_num();
		for (i=0; i < N; i++) {
			printf("Thread ID %d Iter %d\n",id,i);	
		}
	}
	return 0;
}
