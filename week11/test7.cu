#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils/cheader.h"

#define SIZE (2048*2048)

__global__ void add(int *a, int *b, int *c) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	c[i] = a[i] + b[i];
}

void one_thread_add(int *a, int *b, int *c, int size) {
	for (int i = 0; i < size; i++) {
		c[i] = a[i] + b[i];
	}
}

int compare_values(int *a, int *b, int size) {
	int pass = 1;
	for (int i = 0; i < size; i++) {
		if (a[i] != b[i]) {
			printf("a[%i] = %i != b[%i] = %i\n", i, a[i], i, b[i]);
			pass = 0;
		}
	}
	return pass;
}

int main() {
	cudaDeviceProp prop;
	int *a, *b, *c, *d;
	int *d_a, *d_b, *d_c;
	double ms;

	cudaGetDeviceProperties(&prop, 0);
	int threads_x_block = prop.maxThreadsPerBlock;
	printf("THREADS X BLOCK = %i\n", threads_x_block);

	a = (int*) malloc(SIZE * sizeof(int));
	b = (int*) malloc(SIZE * sizeof(int));
	c = (int*) malloc(SIZE * sizeof(int));
	d = (int*) malloc(SIZE * sizeof(int));

	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	fill_array(a, SIZE);
	display_array("a: ", a);
	fill_array(b, SIZE);
	display_array("b: ", b);
	
	start_timer();
	one_thread_add(a, b, d, SIZE);
	ms = stop_timer();
	printf("CPU TIME = %lf\n", ms);
	display_array("CPU c: ", d);
	
	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	start_timer();
	add<<<SIZE/threads_x_block, threads_x_block>>>(d_a, d_b, d_c);
	ms = stop_timer();
	printf("CUDA TIME = %lf\n", ms);

	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("CUDA c: ", c);

	int res = compare_values(c, d, SIZE);
	if (res) {
		printf("all is fine\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);
	free(d);

	return 0;
}

