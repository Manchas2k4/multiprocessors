// =================================================================
//
// File: example4.c
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

#define RENS    3000
#define COLS    3000
#define THREADS 256
#define BLOCKS	((COLS / THREADS) + 1)

__global__ void matrix_vector(int *m, int *b, int *c) {
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int i, sum = 0;

  if (tid < COLS){
    sum = 0;
    for(i = 0; i < RENS; i++) {
          sum += b[i] * m[(i * COLS) + tid];
    }
    c[tid] = sum;
  }
}

int main(int argc, char* argv[]) {
	int i, j, *m, *b, *c;
  int *d_m, *d_b, *d_c;
	double ms;

	m = (int*) malloc(sizeof(int) * RENS* COLS);
	b = (int*) malloc(sizeof(int) * RENS);
	c = (int*) malloc(sizeof(int) * RENS);

  for (i = 0; i < RENS; i++) {
		for (j = 0; j < COLS; j++) {
			m[(i * COLS) + j] = (j + 1);
		}
		b[i] = 1;
	}

  cudaMalloc((void**)&d_m, sizeof(int) * RENS* COLS);
  cudaMalloc((void**)&d_b, sizeof(int) * RENS);
  cudaMalloc((void**)&d_c, sizeof(int) * RENS);

  cudaMemcpy(d_m, m, sizeof(int) * RENS* COLS, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(int) * RENS, cudaMemcpyHostToDevice);

	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();

		matrix_vector<<<BLOCKS, THREADS>>>(d_m, d_b, d_c);

		ms += stop_timer();
	}

  cudaMemcpy(c, d_c, sizeof(int) * RENS, cudaMemcpyDeviceToHost);

	display_array("c:", c);
	printf("avg time = %.5lf ms\n", (ms / N));

  cudaFree(d_m); cudaFree(d_b); cudaFree(d_c);
	free(m); free(b); free(c);
	return 0;
}