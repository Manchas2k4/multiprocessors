// =================================================================
//
// File: example4.c
// Author(s):
// Description: This file contains the code to count the number of
//				even numbers within an array. The time this implementation
//				takes will be used as the basis to calculate the
//				improvement obtained with parallel technologies.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

// array size
#define SIZE 1000000000

// implement your code

int main(int argc, char* argv[]) {
	int i, *a, result;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		// call the implemented function

		ms += stop_timer();
	}
	printf("result = %i\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));
	// must display: result = 500000000

	free(a);
	return 0;
}
