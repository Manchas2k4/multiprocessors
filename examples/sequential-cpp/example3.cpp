/* This code calculates the standard deviation of a set of values */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/cheader.h"

#define SIZE 1000000000

double deviation(int *array, int size) {
	int i;
	double acum, avg;
	
	acum = 0;
	for (i = 0; i < size; i++) {
		acum += array[i];
	}
	avg = acum / size;
	
	acum = 0;
	for (i = 0; i < size; i++) {
		acum = (array[i] - avg) * (array[i] - avg);
	}
	return (sqrt(acum / size));
}

int main(int argc, char* argv[]) {
	int i, j, *array;
	double ms, result;
	
	array = (int *) malloc(sizeof(int) * SIZE);
	random_array(array, SIZE);
	display_array("array", array);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		result = deviation(array, SIZE);
		
		ms += stop_timer();
	}
	printf("S = %.15lf\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));
	return 0;
}
