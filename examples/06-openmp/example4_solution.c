// =================================================================
//
// File: example4.c
// Author: Pedro Perez
// Description: This file contains the code to count the number of
//				even numbers within an array using OpenMP.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define SIZE 1000000000 //1e9

int count_events(int *array, int size) {
	int i, acum;

	acum = 0;
  #pragma omp parallel for shared(array, size) reduction(+:acum)
	for (i = 0; i < size; i++) {
		if (array[i] % 2 == 0) {
			acum++;
		}
	}
	return acum;
}

int main(int argc, char* argv[]) {
	int i, j, *a, result;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = count_events(a, SIZE);

		ms += stop_timer();
	}
	printf("result = %i\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
