#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include "utils.h"

#define SIZE	1e6
#define THREADS	256
#define BLOCKS	MIN(32, (SIZE + THREADS - 1)/ THREADS)

__global__ void minimum(int *array, int *results) {
	__shared__ int cache[THREADS];

	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;

	int aux = 2147483647;
	while (tid < SIZE) {
		aux = (aux < array[tid])? aux : array[tid];
		tid += blockDim.x * gridDim.x;
	}

	cache[cacheIndex] = aux;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i > 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] = (cache[cacheIndex] < cache[cacheIndex + i])? cache[cacheIndex] : cache[cacheIndex + 1];
		}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex == 0) {
		results[blockIdx.x] = cache[cacheIndex];
	}
}

int main(int argc, char* argv[]) {
	int i, *a, *results;
  int *d_a, *d_r;
	double ms;

	a = (int *) malloc(sizeof(int) * SIZE);
	random_array(a, SIZE);
	display_array("a", a);

  results = (int *) malloc(sizeof(int) * BLOCKS);

	cudaMalloc( (void**) &d_a, SIZE * sizeof(int) );
	cudaMalloc( (void**) &d_r, BLOCKS * sizeof(int) );

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	printf("Starting...\n");
	ms = 0;
	for (i = 1; i <= N; i++) {
		start_timer();

		minimum<<<BLOCKS, THREADS>>> (d_a, d_r);

		ms += stop_timer();
	}

	cudaMemcpy(results, d_r, BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);

	int aux = INT_MAX;
	for (i = 0; i < BLOCKS; i++) {
		aux = MIN(aux, results[i]);
	}

	printf("minimum = %i\n", aux);
	printf("avg time = %.5lf\n", (ms / N));

	cudaFree(d_r);
	cudaFree(d_a);

	free(a);
  free(results);
	return 0;
}
