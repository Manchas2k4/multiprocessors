// =================================================================
//
// File: example13.c
// Author: Pedro Perez
// Description: This file implements the quick sort algorithm. The
//				time this implementation takes will be used as the
//				basis to calculate the improvement obtained with
//				parallel technologies.
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

#define SIZE 10000000 //1e7

void swap(int *a, int i, int j) {
	int aux = a[i];
	a[i] = a[j];
	a[j] = aux;
}

int find_pivot(int *A, int low, int high) {
	int i;

	for (i = low + 1; i <= high; i++) {
		if (A[low] > A[i]) {
			return A[low];
		} else if (A[low] < A[i]){
			return A[i];
		}
	}
	return -1;
}

int make_partition(int *a, int low, int high, int pivot) {
	int i, j;

	i = low;
	j = high;
	while (i < j) {
		swap(a, i , j);
		while (a[i] < pivot) {
			i++;
		}
		while (a[j] >= pivot) {
			j--;
		}
	}
	return i;
}

void quick(int *A, int low, int high) {
	int pivot, pos;

	pivot = find_pivot(A, low, high);
	if (pivot != -1) {
		pos = make_partition(A, low, high, pivot);
		quick(A, low, pos - 1);
		quick(A, pos, high);
	}
}

void quick_sort(int *A, int size) {
	quick(A, 0, size - 1);
}

int main(int argc, char* argv[]) {
	int i, j, *a, *aux;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	aux = (int*) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("before", a);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		memcpy(aux, a, sizeof(int) * SIZE);
		quick_sort(aux, SIZE);

		ms += stop_timer();
	}
	display_array("after", aux);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a); free(aux);
	return 0;
}
