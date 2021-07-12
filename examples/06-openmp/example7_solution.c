// =================================================================
//
// File: example7.c
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM using OpenMP.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

#define MAXIMUM 1000000 //1e6

int is_prime(int n) {
	for (int i = 2; i < ((int) sqrt(n)); i++) {
		if (n % i == 0) {
			return 0;
		}
	}
	return 1;
}

void generate_prime(int *arr, int size) {
	int i;

  #pragma omp parallel for shared(arr, size)
	for (i = 2; i < size; i++) {
		arr[i] = is_prime(i);
	}
}

int main(int argc, char* argv[]) {
	int i, *a;
	double ms;

	a = (int *) malloc(sizeof(int) * (MAXIMUM + 1));
	printf("At first, neither is a prime. We will display to TOP_VALUE:\n");
	for (i = 2; i < TOP_VALUE; i++) {
		a[i] = 0;
		printf("%i ", i);
	}
	printf("\n");

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		generate_prime(a, (MAXIMUM + 1));

		ms += stop_timer();
	}
	printf("Expanding the numbers that are prime to TOP_VALUE:\n");
	for (i = 2; i < TOP_VALUE; i++) {
		if (a[i] == 1) {
			printf("%i ", i);
		}
	}
	printf("\n");
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
