#include <stdio.h>
#include <stdlib.h>
#include "utils/cheader.h"

#define SIZE 	1e6
#define THREADS 128

__global__ void add(int *a, int *b, int *c) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < SIZE) {
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[]) {
	int *a, *b, *c, i;
	int *d_a, *d_b, *d_c;
	double ms;
	
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
	
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		add<<<SIZE/THREADS, THREADS>>>(d_a, d_b, d_c);
		ms += stop_timer();
	}
	
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c", c);
	
	printf("avg time = %.15f\n", (ms / N));
	
	cudaFree(d_c);
	cudaFree(d_b);
	cudaFree(d_a);
	
	free(c);
	free(b);
	free(a);
	
	return 0;
}
