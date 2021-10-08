// =================================================================
//
// File: example5.c
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm using OpenMP.
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

void oddEvenSort(int *arr, int size) {
    int step, i, temp;

    #pragma omp parallel shared(arr, size) private(i, temp, step)
    for (step = 0; step < size; step++) {
        if (step % 2 == 0) {
            // even index
            #pragma omp for
            for(i = 0; i <= size - 2; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
            }
        } else {
            // odd index
            #pragma omp for
            for(i = 1; i <= size - 2; i += 2) {
                if (arr[i] > arr[i + 1]) {
                    temp = arr[i];
                    arr[i] = arr[i + 1];
                    arr[i + 1] = temp;
                }
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
		oddEvenSort(aux, SIZE);

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
