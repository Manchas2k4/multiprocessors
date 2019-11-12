#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "utils/cheader.h"

#define SIZE 512

__global__ void add(int *c, int *a, int *b) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

int main(int argc, char* argv[]) {
	int *a, *b, *c;
	int *da, *db, *dc;
	
	a = (int*) malloc(SIZE * sizeof(int));
	fill_array(a, SIZE);
	display_array("a", a);
	
	b = (int*) malloc(SIZE * sizeof(int));
	fill_array(b, SIZE);
	display_array("b", b);
	
	c = (int*) malloc(SIZE * sizeof(int));
	
	double ms = 0;
	start_timer();
	cudaMalloc((void**) &da, SIZE * sizeof(int));
	cudaMalloc((void**) &db, SIZE * sizeof(int));
	cudaMalloc((void**) &dc, SIZE * sizeof(int));
	
	cudaMemcpy(da, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
	
	add<<<SIZE, 1>>>(dc, da, db);
	
	cudaMemcpy(c, dc, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	ms = stop_timer();
	
	display_array("c", c);
	printf("time (ms) = %lf\n", ms);
	
	cudaFree(da); cudaFree(db); cudaFree(dc);
	free(a); free(b); free(c);
	
	return 0;
}
