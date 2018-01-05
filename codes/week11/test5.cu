#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "utils/cheader.h"

#define SIZE (2048*2048)

__global__ void add(int *a, int *b, int *c) {
	int i = blockIdx.x;
	while (i < SIZE) {
		c[i] = a[i] + b[i];
		i += gridDim.x;
	}
}

void fill_array2(int *a, int *b, int size) {
	int i;
	
	for (i = 0; i < size; i++) {
		a[i] = i;
		b[i] = 2 * i;
	}
}

int compare_values(int *a, int size) {
	int pass = 1;
	for (int i = 0; i < size; i++) {
		if (a[i] != (3 * i)) {
			printf("a[%i] = %i != %i\n",i, a[i], (3 * i));
			pass = 0;
		}
	}
	return pass;
}


int main() {
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	double ms;

	a = (int*) malloc(SIZE * sizeof(int));
	b = (int*) malloc(SIZE * sizeof(int));
	c = (int*) malloc(SIZE * sizeof(int));
	
	fill_array2(a, b, SIZE);
	display_array("a: ", a);
	display_array("b: ", b);

	cudaMalloc((void**) &d_a, SIZE * sizeof(int));
	cudaMalloc((void**) &d_b, SIZE * sizeof(int));
	cudaMalloc((void**) &d_c, SIZE * sizeof(int));

	cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);

	start_timer();
	add<<<128, 1>>>(d_a, d_b, d_c);
	ms = stop_timer();

	cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("time = %.5lf\n", ms);
	display_array("c: ", c);

	int res = compare_values(c, SIZE);
	if (res) {
		printf("all is fine\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);

	return 0;
}

