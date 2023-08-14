// =================================================================
//
// File: example5.c
// Author: Pedro Perez
// Description: This file contains the code that implements the
//				bubble sort algorithm using CUDA.
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

#define SIZE 10000
#define THREADS 256

__global__ void even(int* arr, int size) {
  int i, aux;

  i = (threadIdx.x * 2);
  if (i <= size - 2) {
    if (arr[i] > arr[i + 1]) {
      aux = arr[i];
      arr[i] = arr[i + 1];
      arr[i + 1] = aux;
    }
  }
}

__global__ void odd(int* arr, int size) {
  int i, aux;

  i = (threadIdx.x * 2) + 1;
  if (i <= size - 2) {
    if (arr[i] > arr[i + 1]) {
      aux = arr[i];
      arr[i] = arr[i + 1];
      arr[i + 1] = aux;
    }
  }
}

int main(int argc, char* argv[]) {
	int i, j, *a, *d_a;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("a", a);

	cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );

  printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);

		start_timer();

    for (j = 0; j <= SIZE / 2; j++) {
      even<<<1, THREADS>>>(d_a, SIZE);
      odd<<<1, THREADS>>>(d_a,SIZE);
    }

		ms += stop_timer();
	}

  cudaMemcpy(a, d_a, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("a", a);
  printf("avg time = %.5lf ms\n", (ms / N));

	cudaFree(d_a);

	free(a);

  return 0;
}
