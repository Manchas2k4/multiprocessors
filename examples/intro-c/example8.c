// =================================================================
//
// File: example8.c
// Author(s): 
// Description: This file contains the code that implements the
//				enumeration sort algorithm. The time this implementation
//				takes ill be used as the basis to calculate the
//				improvement obtained with parallel technologies.
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

// implement your code

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

		// call the implemented function

		ms += stop_timer();
	}
	display_array("after", a);
	printf("avg time = %.5lf ms\n", (ms / N));

	free(a);
	return 0;
}
