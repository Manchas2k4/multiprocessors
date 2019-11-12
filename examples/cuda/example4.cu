/* This code calculates the standard deviation of a set of values */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "utils/cheader.h"

#define MIN(a,b) (a<b?a:b)

#define SIZE	1e8
#define THREADS	256
#define BLOCKS	MIN(32, (SIZE + THREADS - 1)/ THREADS)

__global__ void add(int *c, int *a, int *b) {
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i < SIZE) {
		c[i] = a[i] + b[i];
	}
}

int main(int argc, char* argv[]) {
	int i, j, *a, *b, *c;
	int *da, *db, *dc;
	double ms;
	
	a = (int *) malloc(sizeof(int) * SIZE);
	fill_array(a, SIZE);
	display_array("a", a);
	
	b = (int *) malloc(sizeof(int) * SIZE);
	fill_array(b, SIZE);
	display_array("b", b);
	
	c = (int *) malloc(sizeof(int) * SIZE);
	
	cudaMalloc((void**) &da, SIZE * sizeof(int));
	cudaMalloc((void**) &db, SIZE * sizeof(int));
	cudaMalloc((void**) &dc, SIZE * sizeof(int));
	
	cudaMemcpy(da, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	printf("Starting...\n");
	ms = 0;
	for (i = 0; i < N; i++) {
		start_timer();
		
		add<<<BLOCKS, THREADS>>>(dc, da, db);
		
		ms += stop_timer();
	}
	
	cudaMemcpy(c, dc, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	display_array("c", c);
	printf("avg time = %.5lf ms\n", (ms / N));
	
	cudaFree(dc);
	cudaFree(db);
	cudaFree(da);
	
	free(a);
	free(b);
	free(c);
	return 0;
}
