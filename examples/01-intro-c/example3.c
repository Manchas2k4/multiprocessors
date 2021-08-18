// =================================================================
//
// File: example3.c
// Author: Pedro Perez
// Description: This file contains the code to perform the numerical
//				integration of a function within a defined interval.
//				The time this implementation takes will be used as
//				the basis to calculate the improvement obtained with
//				parallel technologies.
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

#define PI 3.14159265
#define RECTS 1000000000 //1e9

double function(double x) {
	return sin(x);
}

double integration(double a, double b, double (*fn) (double)) {
	double high, dx, acum, x;
	int i;

	x = MMIN(a, b);
	dx = (MMAX(a, b) - MMIN(a, b)) / RECTS;
	acum = 0;
	for (i = 0; i < RECTS; i++) {
		acum += fn(x + (i * dx));
	}
	acum = acum * dx;
}

int main(int argc, char* argv[]) {
	int i;
	double ms, result;

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		result = integration(0, PI, sin);

		ms += stop_timer();
	}
	printf("sum = %lf\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));

	return 0;
}
