#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define MIN(a,b) (a<b?a:b)

#define SIZE	1e9
#define THREADS	256
#define BLOCKS	MIN(32, (SIZE + THREADS - 1)/ THREADS)

__global__ void dot(float *a, float *b, float *c) {
	__shared__ float cache[THREADS];
	
	int tid = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;
	
	int temp = SIZE;
	while (tid < SIZE) {
		temp = a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	
	cache[cacheIndex] = temp;
	
	__syncthreads();
	
	int i = blockDim.x / 2;
	while (i > 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] = cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0) {
		c[blockIdx.x] = cache[cacheIndex];
	}
}

int main(int argc, char* argv[]) {
	float *a, *b, *c;
	float *d_a, *d_b, *d_c;
	double ms;
	int i;
	
	a = (float*) malloc( SIZE * sizeof(float) );
	b = (float*) malloc( SIZE * sizeof(float) );
	c = (float*) malloc( BLOCKS * sizeof(float) );
	
	for (i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
	
	cudaMalloc( (void**) &d_a, SIZE * sizeof(float) );
	cudaMalloc( (void**) &d_b, SIZE * sizeof(float) );
	cudaMalloc( (void**) &d_c, BLOCKS * sizeof(float) );
	
	cudaMemcpy(d_a, a, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(float), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i <= N; i++) {
		start_timer();
		dot<<<BLOCKS, THREADS>>> (d_a, d_b, d_c);
		ms += stop_timer();
	}
	
	cudaMemcpy(c, d_c, BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);
	
	float acum = 0;
	for (i = 0; i < BLOCKS; i++) {
		acum += c[i];
	}
	
	printf("dot = %f\n", acum);
	printf("avg time = %.5lf\n", (ms / N));
	
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	
	free(c);
	free(b);
	free(a);
	return 0;
}
