// =================================================================
//
// File: example5.c
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm. The time this implementation takes
//				will be used as the basis to calculate the improvement
//				obtained with parallel technologies.
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

void swap(int *A, int i, int j) {
  int aux = A[i];
  A[i] = A[j];
  A[j] = aux;
}

void bubble_sort(int *arr, int size) {
	for(int i = size - 1; i > 0; i--){
		for(int j = 0; j < i; j++){
			if(arr[j] > arr[j + 1]){
				swap(arr, j, j + 1);
			}
		}
	}
}

int main(int argc, char* argv[]) {
	int i, *a, *aux;
	double ms;

	a = (int*) malloc(sizeof(int) * SIZE);
	aux = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		memcpy(aux, a, sizeof(int) * SIZE);
		bubble_sort(aux, SIZE);

		ms += stop_timer();

		if (i == (N - 1)) {
			memcpy(a, aux, sizeof(int) * SIZE);
		}
	}
	display_array("after", a);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a); free(aux);
	return 0;
}
