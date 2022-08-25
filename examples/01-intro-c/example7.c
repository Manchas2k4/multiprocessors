// =================================================================
//
// File: example7.c
// Author(s):
// Description: This file contains the code to brute-force all
//				prime numbers less than MAXIMUM. The time this
//				implementation takes will be used as the basis to
//				calculate the improvement obtained with parallel
//				technologies.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define MAXIMUM 1000001 //1e6

// implement your code

int main(int argc, char* argv[]) {
	int i;
	double ms, result;

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		// call the implemented function

		ms += stop_timer();
	}
	printf("result = %i\n", result);
	printf("avg time = %.5lf ms\n", (ms / N));

	return 0;
}
