/* This code calculates the standard deviation of a set of values */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/cheader.h"

#define MIN(a,b) (a<b?a:b)

#define SIZE	1e9
#define THREADS	256
#define BLOCKS	MIN(32, (SIZE + THREADS - 1)/ THREADS)

__global__ void sum(int *array, long *result) {
	__shared__ long cache[THREADS];
	
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;
	
	long acum = 0;
	while (tid < SIZE) {
		acum += array[tid];
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
		result[blockIdx.x] = cache[cacheIndex];
	}
}

int main(int argc, char* argv[]) {
	int i, *a, *da;
	long acum, *results, *dr;
	double ms;
	
	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);
	
	results = (long*) malloc( BLOCKS * sizeof(long) );
	
	cudaMalloc( (void**) &da, SIZE * sizeof(int) );
	cudaMalloc( (void**) &dr, BLOCKS * sizeof(long) );
	
	cudaMemcpy(da, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		sum<<<BLOCKS, THREADS>>> (da, dr);
		
		ms += stop_timer();
	}
	
	cudaMemcpy(results, dr, BLOCKS * sizeof(long), cudaMemcpyDeviceToHost);
	
	acum = 0;
	for (i = 0; i < BLOCKS; i++) {
		acum += results[i];
	}
	
	printf("sum = %li\n", acum);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	cudaFree(dr);
	cudaFree(da);
	
	free(a);
	return 0;
}
