#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/cheader.h"

#define SIZE 8300000

__global__ void squares(int *a, int *b) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	b[i] = a[i] * a[i];
}

int main(int argc, char* argv[]) {
	int *a;
	int *d_a, *d_b;
	double ms;
	
	a = (int*) malloc(SIZE * sizeof(int));
	fill_array(a, SIZE);
	display_array("before:", a);
	
	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (int i = 0; i < N; i++) {
		start_timer();
		squares<<<SIZE/128, 128>>>(d_a, d_b);
		ms += stop_timer();
	}
	
	cudaMemcpy(a, d_b, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("after: ", a);
	printf("avg time O %.5lf ms\n", (ms/N) );
	
	cudaFree(d_a);
	cudaFree(d_b);
	
	free(a);
	
	return 0;
}
