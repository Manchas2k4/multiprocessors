/* This code adds all the values of an array */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/cheader.h"

#define SIZE 1000000000

long sum(int* array, int size) {
	int i;
	long acum = 0;

	for (i = 0; i < size; i++) {
		acum += *(array + i);
	}
	return acum;
}

int main(int argc, char* argv[]) {
	int i, *array, j;
	double ms;
	long r;

	array = (int*) malloc(sizeof(int) * SIZE);
	fill_array(array, SIZE);
	display_array("array", array);
	
	printf("Starting...\n");
	ms = 0;
	for (j = 0; j <= N; j++) {
		start_timer();
		r = sum(array, SIZE);
		ms += stop_timer();
	}
	printf("sum = %li\n", r);
	printf("avg time = %.5lf\n", (ms/N));
	
	free(array);
	return 0;
}
