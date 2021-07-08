#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define SIZE	1e6
#define THREADS	256
#define BLOCKS MIN(32, (SIZE + THREADS - 1)/ THREADS)

#define PI		3.14159265
#define RECTS 1000000 //1e6
#define START 0.0
#define END 	PI

__global__ void integration(double *x, double *dx, double *results) {
	__shared__ double cache[THREADS];

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;

	double acum = 0;
	while (tid < SIZE) {
    acum += sin( (*x) + (tid * (*dx)) );
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = acum;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i > 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		results[blockIdx.x] = cache[cacheIndex];
	}
}

int main(int argc, char* argv[]) {
	double x, dx, *results;
  double *d_x, *d_dx, *d_r;
	double ms;
  int i;

  x = START;
  dx = (END - START) / RECTS;

	results = (double*) malloc( BLOCKS * sizeof(double) );

	cudaMalloc( (void**) &d_x, sizeof(double));
  cudaMalloc( (void**) &d_dx, sizeof(double));
	cudaMalloc( (void**) &d_r, BLOCKS * sizeof(double) );

	cudaMemcpy(d_x, &x, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dx, &dx, sizeof(double), cudaMemcpyHostToDevice);

	printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
		start_timer();

		integration<<<BLOCKS, THREADS>>> (d_x, d_dx, d_r);

		ms += stop_timer();
	}

	cudaMemcpy(results, d_r, BLOCKS * sizeof(double), cudaMemcpyDeviceToHost);

	double acum = 0;
	for (i = 0; i < BLOCKS; i++) {
		acum += results[i];
	}

	printf("area = %.5lf\n", (acum * dx));
	printf("avg time = %.5lf\n", (ms / N));

  cudaFree(d_x);
  cudaFree(d_dx);
	cudaFree(d_r);

	free(results);
	return 0;
}
