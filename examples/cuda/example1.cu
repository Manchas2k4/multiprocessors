/* This code calculates an IP approximation */
#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define MIN(a,b) (a<b?a:b)

#define NUM_RECTS 1e9
#define THREADS	256
#define BLOCKS	MIN(32, (NUM_RECTS + THREADS - 1)/ THREADS)

__global__ void kernel(float width, float *results) {
	__shared__ long cache[THREADS];
	int i, cacheIndex;
	float acum, mid, height;
	
	i = threadIdx.x + (blockIdx.x * blockDim.x);
	cacheIndex = threadIdx.x;
	
	acum = 0;
	while (i < NUM_RECTS) {
		mid = (i + 0.5) * width;
		height = 4.0 / (1.0 + (mid * mid));
		acum += height;
			
		i += blockDim.x * gridDim.x;
	}
	
	cache[cacheIndex] = acum;
	
	__syncthreads();
	
	i = blockDim.x / 2;
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
	float acum, width, area;
	float *results, *dr;
	double ms;
	int i;
	
	width = 1.0 / (double) NUM_RECTS;
	
	results = (float*) malloc( BLOCKS * sizeof(float) );
	
	cudaMalloc( (void**) &dr, BLOCKS * sizeof(float) );
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		kernel<<<BLOCKS, THREADS>>> (width, dr);
		
		ms += stop_timer();
	}
	
	cudaMemcpy(results, dr, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
	
	acum = 0;
	for (i = 0; i < BLOCKS; i++) {
		acum += results[i];
	}
	area = width * acum;
	
	printf("PI = %.15lf\n", area);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	cudaFree(dr);
	
	free(results);
	return 0;
}
