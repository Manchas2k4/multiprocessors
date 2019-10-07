#include <stdio.h>
#include <stdlib.h>
//#include <math.h>
#include "utils/cheader.h"
#include <omp.h>

#define SIZE 1000000000

double sum_array(int *array, int size) {
	double acum = 0;
	int i;
	
	#pragma omp parallel for shared(array, size) reduction(+: acum)
	for (i = 0; i < size; i++) {
		acum += array[i];
	}
	return acum;
}

int main(int argc, char* argv[]) {
	int i, j, *a;
	double ms, result;
	
	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		result = sum_array(a, SIZE);
		
		ms += stop_timer();
	}
	printf("sum = %lf\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	free(a);
	return 0;
}
