// =================================================================
//
// File: example6.c
// Author: Pedro Perez
// Description: This file implements the multiplication of a matrix
//				by a vector. The time this implementation takes will
//				be used as the basis to calculate the improvement
//				obtained with parallel technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "utils.h"

#define RENS 30000
#define COLS 30000

void matrix_vector(int *m, int *b, int *c) {
	int i, j, acum;

	for (i = 0; i < RENS; i++) {
		acum = 0;
		for (j = 0; j < COLS; j++) {
			acum += (m[(i * COLS) + j] * b[i]);
		}
		c[i] = acum;
	}
}

int main(int argc, char* argv[]) {
	int i, j, *m, *b, *c;
	double ms;

	m = (int*) malloc(sizeof(int) * RENS * COLS);
	b = (int*) malloc(sizeof(int) * RENS);
	c = (int*) malloc(sizeof(int) * RENS);

	for (i = 0; i < RENS; i++) {
		for (j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		matrix_vector(m, b, c);

		ms += stop_timer();
	}
	display_array("c:", c);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(m); free(b); free(c);
	return 0;
}
