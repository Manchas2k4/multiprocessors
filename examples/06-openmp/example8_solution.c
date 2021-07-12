// =================================================================
//
// File: example7.c
// Author(s):
// Description: This file contains the code that implements the
//				enumeration sort algorithm using OpenMP.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

#define SIZE 10000

void enumeration_sort(int *array, int size, int copy) {
	int *temp = (int *) malloc(sizeof(int) * size);
	int i, j, count;

  #pragma omp parallel for shared(array, temp, size) private(j, count)
	for (i = 0; i < size; i++) {
		count = 0;
		for (j = 0; j < size; j++) {
			if (array[j] < array[i]) {
				count++;
			} else if (array[i] == array[j] && j < i) {
				count++;
			}
		}
		temp[count] = array[i];
	}
	if (copy) {
		memcpy(array, temp, sizeof(int) * size);
	}
	free(temp);
}

int main(int argc, char* argv[]) {
	int i, *a;
	double ms;

	a = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		enumeration_sort(a, SIZE, (i == (N - 1)));

		ms += stop_timer();
	}
	display_array("after", a);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
