#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/cheader.h"

#define SIZE 8300000

__global__ void add(int *c, int *a, int *b) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	c[i] = a[i] + b[i];
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	double ms;
	
	a = (int*) malloc(SIZE * sizeof(int));
	b = (int*) malloc(SIZE * sizeof(int));
	c = (int*) malloc(SIZE * sizeof(int));
	
	fill_array(a, SIZE);
	display_array("a:", a);
	fill_array(b, SIZE);
	display_array("b:", b);
	
	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));
	
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();
		add<<<SIZE/128, 128>>>(d_c, d_a, d_b);
		ms += stop_timer();
	}
	
	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c:", c);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	free(a);
	free(b);
	free(c);
	
	return 0;
}
	
