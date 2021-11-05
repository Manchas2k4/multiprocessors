// =================================================================
//
// File: intro4.cu
// Author: Pedro Perez
// Description: This file shows some of the basic CUDA directives.
//
// Copyright (c) 2020 by Tecnologico de Monterrey.
// All Rights Reserved. May be reproduced for any non-commercial
// purpose.
//
// =================================================================

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils.h"

#define SIZE 512

__global__ void add(int *a, int *b, int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	a = (int*) malloc(SIZE * sizeof(int));
	fill_array(a, SIZE);
	display_array("a", a);

	b = (int*) malloc(SIZE * sizeof(int));
	fill_array(b, SIZE);
	display_array("b", b);

	c = (int*) malloc(SIZE * sizeof(int));

	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	add<<<SIZE, 1>>>(d_a, d_b, d_c);

	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c", c);

	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);

	free(c);
	free(b);
	free(a);

	return 0;
}
