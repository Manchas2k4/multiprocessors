#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define SIZE 100000000

#define imin(a,b) (a<b?a:b)

const int threadsPerBlock = 256;
const int blocksPerGrid = imin(256, (SIZE + threadsPerBlock - 1) / threadsPerBlock);

__global__ void sum(int *a, int *c) {
	__shared__ int cache[threadsPerBlock];
	
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int cacheIndex = threadIdx.x;
	
	long acum = 0;
	while (i < SIZE) {
		acum += a[i];
		i += blockDim.x * gridDim.x;
	}
	cache[cacheIndex] = acum;
	
	__syncthreads();
	
	i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i) {
			cache[cacheIndex] += cache[cacheIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	
	if (cacheIndex == 0) {
		c[blockIdx.x] = cache[0];
	}
}

int main(int argc, char* argv[]) {
	int *a, *c;
	int *d_a, *d_c;
	double ms;
	
	a = (int*) malloc(SIZE * sizeof(int));
	c = (int*) malloc(blocksPerGrid * sizeof(int));
	
	fill_array(a, SIZE);
	display_array("a:", a);
	
	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, blocksPerGrid * sizeof(int));
	
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();
		sum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c);
		ms += stop_timer();
	}
	
	cudaMemcpy(c, d_c, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);
	
	long acum = 0;
	for (int i = 0; i < blocksPerGrid; i++) {
		acum += c[i];
	}
	printf("avg time = %.5lf ms\n", (ms / N));
	printf("sum = %li\n", acum);
	
	cudaFree(d_a);
	cudaFree(d_c);
	
	free(a);
	free(c);
	
	return 0;
}
